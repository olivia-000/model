# 決策與歷程記錄

## 2026-09-08 — HIMLoco checkpoint 架構確認 + IsaacLab 驗證

在 IsaacLab/PhysX 裡測試 `checkpoints/low_level_policies/user_gait_0908/policy.pt`，目的是
為了替移植到 Isaac Gym 做準備（見 `FINDINGS.md`，這份是當時整理帶去另一台機器的完整交接筆記，
建議直接看該檔案原文，這裡只記重點結論）：

- 這顆 checkpoint 是 `PolicyExporterHIM`（HIM = Hybrid Internal Model），不是單純 actor：
  estimator 吃 270 維（45 維 obs × 6 步歷史，time-major 排列）算出速度估計 + 16 維 latent，
  actor 再吃「最新一步 45 維 + 這兩個估計值」= 64 維輸出動作。**不要**自己拆開 estimator/actor
  分開重寫，整包當黑盒子丟 270 維進去就對了。
- 在 IsaacLab 裡要跟原始 legged_gym 訓練時一致，三個致命參數必須對齊，對錯任何一個都會導致
  「看起來wiring正確但高速就垮」：`stiffness=40/damping=1`（不是 IsaacLab 預設的 25/0.5）、
  `torque_limit=33.5`（不是預設 23.5）、且致動器模型必須是**平頭夾限 PD**，不能是 IsaacLab
  預設的 DC-motor 速度相關飽和曲線（換算成同樣數值也沒用，模型本身不同，這點是實測花時間才
  確認的，不是猜測）。
- 在 IsaacLab 裡速度越快越不穩（vx=0.6 直接反覆崩潰），懷疑是 PhysX 的 contact/friction
  solver 跟原始訓練環境（大概率是 Isaac Gym/legged_gym 本身）有 sim-to-sim 落差 —— 這正是
  「拿去 Isaac Gym 上驗證同一顆 checkpoint 在高速還穩不穩」這件事本身的動機。

## 2026-09-09 — Isaac Gym 完整整合（play_stairs_viplanner.py）

在 `/home/itri/HIMLoco/legged_gym/` 這個 legged_gym repo 裡，把上面驗證過的 checkpoint
接上 ViPlanner 視覺導航，實際在 `go2_stairs` 任務（U 型雙截樓梯 + 2F 房間）裡跑通。

### 為什麼拿掉 PathFollower，改成直接用相機座標軌跡點轉向
消融實驗 `play_stairs_viplanner_minimal.py`（拿掉整個拓樸層/PathFollower，單一固定目標點）
爬得比原本這支腳本（有 PathFollower）高很多：z 從 0.42 爬到 1.17，而原本版本卡在底部附近。
heading error 平均只有 0.09 度（~600 次 replan），證明 ViPlanner 自己的輸出方向沒問題，
問題出在 PathFollower 的 lookahead 搜尋 + world-frame 來回轉換 + 過高的 yaw gain。所以後來
把「直接讀相機座標系軌跡上第 15 個點算 heading_error = atan2(py, px)」直接搬進正式版，
PathFollower 整層拔掉。拓樸層（MultiFloorGoalManager/SubgoalSequencer，決定「現在該往哪個
subgoal 走」）保留不變。

### 為什麼指令改成「每 2 秒才更新一次 + 15 度死區」
這顆 checkpoint 本身有天生 yaw drift（`test_inherent_yaw_drift.py` 證實：即使指令是
「完全不轉」wz=0，機器人自己還是會固定往一個方向偏 10-22 度 —— 這是步態本身的不對稱，不是
控制邏輯的 bug）。原本每次 replan（~0.1秒）都送一次新指令，還加了 EMA 平滑（alpha=0.3）想
讓動作更穩，結果平滑把 70% 的修正指令都濾掉了，天生漂移就贏了，反而讓機器人卡住不動。拿掉
平滑後改善但還沒到最好。真正找到答案是對照 `play_stairs.py`（wz 永遠鎖 0，什麼都不修正）
反而爬得更高（z 到 1.16-1.84）——差別不在數值大小，而在**指令改變的頻率**：訓練時
`commands.resampling_time=10.0`，指令十秒才變一次，這支腳本卻每 0.1 秒就變一次，是訓練時
從沒見過的指令變化率。所以最終方案：vision 還是每 `REPLAN_EVERY_N_STEPS` 算一次估計值，
但真正送給機器人的指令 `(vx, wz)` 每 `COMMAND_HOLD_STEPS`（~2秒）才「決定」一次，且
heading error 在 15 度死區內就直接鎖 wz=0（不做小幅度連續修正），刻意貼近訓練時「指令很少變」
的分布。

### fear 值為什麼沒接安全門檻
這顆 checkpoint 是用真實世界資料訓練的，對這裡的合成方塊樓梯場景幾乎持續回報 fear≈0.97，
如果真的拿 fear>threshold 當急停條件，機器人會永遠卡住不動。所以目前只把 fear 記錄下來看，
不拿來做控制決策。

## 2026-09-10 — Isaac Sim 版本（獨立 Docker 環境，另開一條線）

同一天另外在 Isaac Sim 5.1.0 裡也把這顆 HIMLoco+ViPlanner 組合跑起來，但這是**完全獨立的
第二套實作**，不是沿用 `play_stairs_viplanner.py`：

- **為什麼另外建一個全新 Docker 環境，而不是用機器上既有的 `isaac-lab-base` container**：
  用戶明確給過最高等級規定——不准碰任何 session 開始前就已經存在的東西（container、
  docker-compose 設定、已安裝軟體等），即使改動看起來很小很安全也一樣。因此才有
  `/home/itri/isaac-sim-docker/`（image `isaac-sim-runtime:5.1.0`）這個全新環境，跟
  `isaac-lab-base` 完全不相干、互不影響。
- **場景跟狗模型都換了**：沒有用 `go2_stairs` 那個程式化樓梯場景，改用一般房間場景
  `Collected_lab_main/lab_main.usd`；狗也沒用 legged_gym 那支 `go2.urdf` 轉出來的模型，
  改用現成的 `/home/itri/fire_dog/go2.usd`。目前這條線**還沒**在樓梯場景裡驗證過。
- **`viplanner_bridge_isaacsim.py` / `camera_semantics_lab.py` 是重新移植的版本**：
  原本 `viplanner_bridge.py`/`camera_semantics.py` 依賴 Isaac Gym 的 API，Isaac Sim 環境
  沒有這些，所以重寫成不依賴 Isaac Gym 的版本；`camera_semantics_lab.py` 的語意分類邏輯也
  改成針對這個房間場景自己的 Wall/Cabinet/Floor bounding box，而不是樓梯場景的
  `ramp_cfg.yml`。
- **踩到的坑（花了約1小時才查出來，之後改這支控制迴圈務必保持順序）**：`viplanner_control.py`
  的 `run_walk()` 裡，PyTorch 絕對不能在 `omni.timeline...play()` 真正跑起來、且模擬跑過
  幾個 app-update 之後，之前碰 CUDA（連 `torch.cuda.is_available()` 都不行）——碰了的話
  第一個 physics step 會無聲無息卡死（沒有任何 log、CPU 卡在 100%）。這是 PhysX 跟 PyTorch
  的 CUDA context 衝突，不是 physics cooking 或 shader 編譯的問題（兩者都各自獨立測過，
  都在 1 秒內就完成，排除嫌疑）。所以目前程式碼順序刻意是：先建 physics scene → 呼叫
  `timeline.play()` → 跑幾個 warm-up update → 再建 Articulation/Camera → 最後才碰任何
  torch/CUDA/ViPlanner 相關程式碼。改這支檔案時不要打亂這個順序。
- **對本專案資料夾的直接影響**：`viplanner_core/viplanner/config/costmap_cfg.py` 和
  `learning_cfg.py` 這兩個檔案被直接改過——dataclass 裡原本用 mutable default（例如
  `field: dict = {...}` 這種寫法）在 Python 3.11 下會直接報錯，改成
  `field(default_factory=...)`。原因是這個包當初是寫給 Python 3.7-3.9 的，Isaac Sim
  這個 docker 環境跑的是 3.11。這兩個檔案現在看到的內容**已經是修好的版本**。

## 2026-09-16 — 樓梯入口「不敢轉彎、繞一大圈」的除錯過程

機器狗接近樓梯 A 入口時，會放棄直線走向入口，改繞一個很大的圈子才（有時）進去，用戶一開始
描述成「不敢轉彎到樓梯」。這是一次先後嘗試了三個假設、前兩個都被實測數據推翻的除錯過程，
記下來是因為結論違反直覺：**表面上看起來像轉向/安全邊界問題，實際根本原因跟牆壁安全邊界
完全無關**。

### 假設一（錯）：身體對角線安全邊界太保守，把整條走廊都算成太靠近牆
`viplanner_control.py` 的 wall-repulsion 邏輯原本對所有 `SEG_ID_WALL` 類物件統一套用
`GO2_HALF_DIAGONAL`（身體半對角線，任何朝向都成立的保守值，約 0.5m 有效邊界）。1.5m 寬的
樓梯走廊兩側都是護欄（薄扶手，真正危險性低），套用這個保守邊界後幾乎整條走廊都會觸發
「離牆太近」。第一次修正：把牆分成「護欄」（護欄/landing 分隔牆）跟「真牆」（房間牆/門框/
邊界牆）兩類，護欄改用方向性投影（沿當下朝向算，比較不保守），後來甚至改成護欄完全跳過
這層安全邊界（USD 實體碰撞還在，只是拿掉這層「軟性」提前迴避）。**結果：用戶回報「還是
一樣」。**

### 驗證假設一是否真的無關：加診斷 log 直接印出每次匹配到的牆是不是護欄
沒有再憑感覺改參數，而是在 wall-repulsion 觸發點加一行 log，印出匹配到的牆是不是護欄、
安全邊界有沒有真的被跳過。實測整段入口逼近路徑上，**每一次匹配到的牆都正確判定成護欄、
安全邊界確實被跳過**，但機器狗還是繞了一大圈。這證明了假設一从頭到尾就是錯的——問題完全
不在牆壁安全邊界這層。這步很關鍵：如果沒有加診斷直接驗證，很容易一直在同一個錯誤方向上
反覆調參數（事實上前兩次「改邊界公式」「乾脆跳過」都沒用，就是因為根本沒對到問題）。

### 假設二（對，但實作有 bug）：入口 waypoint 跟地面同高，樓梯導引機制沒有機會介入
場景裡樓梯導引的機制（一發現「當前 waypoint 的 z 跟機器人現在的 z 差很多」就直接用幾何方式
算朝向、完全不信任 ViPlanner 自己的原始軌跡預測）只有在 `z_gap > 0.3` 時才會啟動。查
`open_scene_stairs.py` 的 waypoint 定義才發現：入口 waypoint 本身就設在「樓梯前方 0.2m、
z=0.35」——幾乎跟地面同高，整段逼近路徑上 `z_gap` 永遠不會超過 0.3。也就是說，在機器狗真正
開始爬樓梯之前，這整段完全交給 ViPlanner 自己的原始預測在決定要往哪走，而它自己就是不想
直線靠近樓梯（可能是訓練資料裡樓梯本身帶有一定 cost，或訓練沒充分覆蓋「正對樓梯直線接近」
這個角度）。加了一個新條件：只要「當前 waypoint 本身」離某個樓梯 footprint 很近（1.5m 內），
不管有沒有高度差，一樣觸發幾何導引。**但第一次實作漏改了一個既有的 `not on_floor_reaching_goal`
guard**（這個 guard 原本是用來過濾「純粹因為 waypoint 標記本身的裝飾性高度誤差造成的假觸發」，
但機器狗本來就站在正確樓層上，這個 guard 幾乎永遠是 True，把新加的判斷整個蓋掉了，等於完全
沒生效）。同樣是先加 log 印出新判斷的值才抓到——印出來發現全程都是 False，才回頭發現是
guard 邏輯的問題，不是判斷本身的距離門檻設錯。

### 假設二實作 bug 修好之後又冒出來的新問題：目標點在兩種選擇策略之間亂跳
guard 修好、機器狗真的會被幾何導引拉向入口之後，測試發現又卡在門檻前原地繞：`dist_to_goal`
在 1.26～2.3 公尺之間震盪超過 3000 步不收斂。根本原因是機器狗的 XY 位置剛好卡在第一階樓梯
AABB 的邊界上，`on_stairs` 這個布林值每次 replan 之間 True/False 來回跳，而目標點選擇邏輯
剛好用 `on_stairs` 分兩套完全不同的策略（已經在爬樓梯時選「下一階」；還沒上樓梯時選「離
機器人當下位置最近的任一階」），兩套策略解出來的目標點常常不是同一階樓梯，導致瞄準方向
每次 replan 都跳來跳去。修法：入口逼近（尚未真正踏上樓梯）這個情境不要再從樓梯幾何重新
推導目標，直接瞄準 waypoint 本身（反正 waypoint 就設在入口前 0.2m），完全繞開會亂跳的
選擇邏輯。修好後連續跑了 7 個 episode 都順利通過入口、爬完整段樓梯。

### 這次除錯的方法論教訓
三個假設裡，只有第一個是靠「加診斷 log 直接驗證」才被推翻的；第二、三個問題也都是靠印出
中間變數的實際值才抓到（分別是 `on_floor_reaching_goal` guard 蓋掉判斷、`on_stairs` 邊界
抖動）。單純看代碼邏輯推理容易漏掉這種「條件組合出來剛好互相抵消」或「邊界值抖動」的情況，
之後遇到類似「看起來合理但實測不符預期」的行為，直接加 log 印真實運行時的中間值，比反覆
猜測調參數快得多。

## 2026-09-22 — L 型單轉角樓梯場景：接上狗 + ViPlanner，一連串除錯

新場景 `l_shaped_stair_scene_with_walls_viplanner.py`（專案根目錄）：把使用者自建的
`l_shaped_stair_scene_with_walls.py`（`FixedCuboid`版，無狗無控制）的座標數學搬到
`UsdGeom.Cube + UsdPhysics.CollisionAPI`，逐箱標 `SEG_ID_FLOOR/STAIRS/WALL`，加狗、算
waypoint、呼叫既有的 `isaac_sim_bridge/viplanner_control.py::run_walk()`（沿用 v3 模型 +
HIMLoco checkpoint）。單轉角樓梯：1.5m 寬、N=6 階、landing 1.5x1.5m、TOP_Z=1.20m。
`aabbs.json` 寫到獨立的 `training_data/l_shaped_stairs_scene/`，不覆蓋主線雙截樓梯自己的
`training_data/stairs_scene/`。手動組的 docker run 指令要記得帶上
`isaac-sim-docker/home:/home/isaacsim:rw`（HOME 掛載，沒這個 texture cache 寫入會失敗導致
整個啟動失敗）、`fire_dog:ro`、`viplanner_ckpt:ro`、`training_data:rw`。

### Bug 1：Stair B（轉 90 度沿 +Y 爬升那段）從側面摔下去
`viplanner_control.py` 的「虛擬邊界牆」（防止機器狗踩空側邊摔下樓梯）原本用「相鄰階梯是否
共用同一個 Y 座標」判斷是不是同一段樓梯——主線雙截樓梯兩段都沿 X 方向，這個假設一直成立。
這個新場景的 Stair B 沿 +Y 爬升，每一階 Y 座標都不同，被誤判成 6 段各自獨立的單階樓梯，
左右安全邊界完全沒作用。爬到 2/3 高度時從側面摔下去，摔回一樓後再也找不回樓梯，超時失敗。

用戶明確要求「樓梯不可以有牆壁」（比照真機沒有護欄），所以沒有在場景裡加實體護欄，改成
直接修 `viplanner_control.py`：用「階梯 AABB 是否真的緊鄰（重疊要達到較窄那個 box 寬度的
至少 50%，不只是 `_AABB_MARGIN` 造成的角落重疊）」做 union-find 分群，再用每群的 X/Y 延伸
長度自動判斷這段樓梯是沿 X 還是沿 Y 爬升，據此算出正確的左右安全邊界。
（第一版分群邏輯只看 AABB 有無重疊，導致 Stair A 和 Stair B 因為兩者角落 boxes 的
`_AABB_MARGIN` padding 剛好重疊 0.06m 被誤判成同一段樓梯——改成上面「50% 寬度重疊」才修對。）

### Bug 2：走去一樓樓梯口途中卡在 Stair A 最高一階附近原地震盪
拿掉護欄、修好分群邏輯後，狗改在「往 waypoint 1（一樓樓梯口）前進」階段卡住——牠從東側遠角
出發，直線路徑會先經過 Stair A 最後一階（最高、離牆很近）附近，那裡有語意補丁
（`_tag_wall_region`，標記「這面其實是牆不是樓梯」），牠被這塊補丁的軟性避牆推力和導航
目標的拉力卡在原地反覆震盪。用戶指出通用原則：「樓梯要在它可以上的高度才可以走，前方標記
是樓梯但高度太高要繞過去」——查到 `viplanner_control.py` 其實已有同樣訴求的舊功能
（`_CLIMB_RISE_LIMIT`，選瞄準哪個樓梯 footprint 時只挑爬得上去的），但只在「z 落差夠大」
那個分支生效，waypoint 1 走的是另一分支（`near_stairs_entrance`，直接瞄準 waypoint 本身，
沒有高度過濾，過去為了修另一個震盪 bug 刻意這樣分開）。修法：新增一段疊加在任何分支選定的
瞄準點之上的「側向閃避」——每步找離狗目前位置最近、但高度超過可爬範圍（0.35m）的樓梯
footprint，若在 1.2m 內就把瞄準點往旁邊推開（不是整個換一套選點策略，避免重蹈舊震盪回歸）。

驗證：4-waypoint 版本連續 3 個 episode 全部成功走到二樓目標（第 8131/8039/8041 步）。

### Bug 3：RViz 觀察到樓梯側面被標記成綠色（可走），逼它繞更大圈
用戶即時看 RViz 語意圖，發現從側面靠近樓梯時，樓梯的左右側面（每一階 box 是實心方塊，側面
其實是垂直牆面，不是能爬的踏面）仍是 `SEG_ID_STAIRS` 綠色，跟真正可走的踏面同色，網路自己
的路徑預測分不出「這個綠色是可爬的」還是「這個綠色只是同一個 class 但角度不對」。
`viplanner_control.py` 的虛擬邊界牆只影響控制迴圈的軟性安全邊界，刻意不寫進 `aabbs`（不
改變網路自己看到的語意圖）——這正是問題所在：物理防摔有了，但感知端沒有对应的修正。
修法：在場景腳本裡對 Stair A / Stair B 各自的左右側面加**語意專用**（無碰撞，複用
`_tag_wall_region`）WALL 補丁，整段長度、固定高度（避免鋸齒輪廓的舊 bug）。加上後 4-waypoint
版本從 ~8100 步大幅加速到 4419 步（幾乎快一倍）。

### Bug 4：用戶要求「上不去的地方都要改成牆壁」，樓梯最高一階背面補丁的間隙太寬
用戶指出樓梯最高一階朝 landing 方向外露的那面（Bug 2 提到的補丁）留的安全間隙太寬
（原本 2 階高度 ≈0.2m，讓機器狗有時會摔在樓梯轉平台的交界處），把間隙縮小成只留 1 階
（≈0.1m），兩段樓梯都改。驗證：4-waypoint 版本第 4696 步到達，z 軌跡平滑爬升無摔落。

### 單一目標點模式（L_SHAPED_SINGLE_GOAL）：從只給 4 waypoint 改成完全自主找路
用戶要求測試「只給二樓目標點，不給任何中間 waypoint，讓機器狗自己判斷 Z 軸落差、自己找
樓梯口」。第一次測試（沿用遠角出生點 (6,5)）第 9819 步成功，但比 4-waypoint 版本慢，且
之後用戶要求把出生點改到「二樓目標點正下方、Z=0」（讓 XY 距離目標幾乎為零，唯一真正的路
要繞一大圈走樓梯間），並把 waypoint 永久改成只有二樓那一個（不再是切換式的環境變數）。

改出生點後接連暴露出三個新 bug（全部是「舊機制隱含假設『出生點在樓梯西側、遠離二樓正下方』
這個前提，被新出生點打破」造成的）：

1. **`_stair_a_cleared` latch 提前誤觸發，一開場就把 Stair A 標成牆**：為了讓「已爬過的
   Stair A 不會被拉回去」這件事在沒有 waypoint index 可判斷的單一目標點模式下也成立，加了
   一個「機器狗 X 座標若曾經超過 Stair A 自己的最大 X」就永久鎖定「Stair A 已清空」的機制。
   但新出生點的 X 本來就在 Stair A 的東側（超過那個門檻），第一步就滿足條件，Stair A 從
   開場就被標成牆——用戶直接在 RViz 上看到、問「為什麼把一樓樓梯標記成牆」。修法：加一個
   `_stair_a_ever_climbed` 旗標，只有機器狗真的進過 Stair A 的 XY 範圍**而且**高度明顯高於
   正常站立基準（>0.45m，站在平地基準約 0.28-0.35m）才算「真的爬過」，latch 才准生效。

2. **`on_floor_reaching_goal` 只比對目標高度、沒比對機器狗自己的高度**：機器狗一開始完全
   不去找樓梯（`seeking_stairs` 全程 False），用戶反映「它應該要去找樓梯口啊」。查出
   `on_floor_reaching_goal`（判斷「目前站的地板本身高度範圍已經涵蓋目標高度，不需要爬樓梯」）
   只檢查目標的 Z 是否落在地板 tag 的 Z 範圍 ± 0.8m margin 內，從來沒檢查機器狗**自己**的
   Z——新出生點的 XY 剛好落在二樓地板（Floor_2）的 XY 範圍正下方，即使人站在一樓地面
   （z=0.29，離 Floor_2 實際表面 1.2m 遠、中間隔著一整片實體樓板），這個判斷依然因為 XY
   對得上而誤判為 True。修法：额外要求機器狗自己的 z 也要落在該地板 tag 的 Z 範圍內——但
   第一版沿用同一個 0.8m margin 還是不夠緊（0.29 仍落在 Floor_2 的 [1.05-0.8, 1.23+0.8]=
   [0.25,2.03] 內），改用不對稱的更緊 margin（下緣只留 0.15m 吸收 AABB padding 誤差、
   上緣留 0.5m 涵蓋機器狗站立高度+ waypoint 標記的 cosmetic 偏移）才真正排除這個假陽性。

3. **entrance-cluster 選點邏輯在兩段樓梯的入口之間來回拉扯 / 爬到一半被拉回自己樓梯的入口**：
   為了讓機器狗能自己找到正確入口（不是隨便選「XY 最近的某一階」，見 Bug 2 之後延伸出的
   `_CLIMB_RISE_LIMIT` 那條線），改成把目前有效的樓梯 AABB 依連通性分群成一段段樓梯，每段
   取「高度最低的那一階」當作該段的天然入口，取離機器狗最近的入口當目標。這個邏輯本身
   後來又暴露兩個子問題：
   (a) 已經在爬某一段樓梯、但暫時找不到「更前面」的下一階時（`ahead` 為空），這套入口選點
       邏輯會被觸發，而機器狗自己所在那段樓梯的入口往往正是離它「最近」的入口，於是被拉回
       自己剛爬上來的樓梯底部，反覆「爬一半、退回底部」循環好幾輪才碰運氣脫困。修法：入口
       選點只在完全不在任何樓梯上（`not on_stairs`）時才觸發；`on_stairs` 但找不到下一階時，
       改用原本的「附近爬得上去的最近一階」邏輯（比較不會跳到很遠的別段樓梯入口）。
   (b) 入口選點沒有做可爬性過濾：從地面某些位置看，Stair B 自己的入口（第一階，高度 0.70m）
       在 XY 直線距離上可能比 Stair A 的入口（第一階，高度 0.10m，真正從地面可爬）更近，
       即使 Stair B 的入口其實得先爬完 Stair A、穿過 landing 才碰得到。機器狗因此在兩段
       樓梯的入口之間來回拉扯，整個 13000 步都卡住沒進展。修法：入口候選也套用跟
       `_CLIMB_RISE_LIMIT` 一樣的可爬性過濾（高度需在機器狗目前高度 +0.35m 內），過濾後
       再挑最近的。

全部三個修好後，單一目標點模式第 3767 步順利到達目標，z 軌跡從頭到尾單調爬升，完全沒有
再摔落或來回拉扯。

### 這次的方法論教訓（延續上面 09-16 那次的心得）
好幾個 bug 都是「舊機制當初的設計假設，被之後的新需求悄悄打破」造成的：
`_stair_a_cleared` 假設出生點在樓梯西側、`on_floor_reaching_goal` 假設不同樓層的地板 XY
不會互相重疊、entrance-cluster 選點假設「最近的入口=可爬的入口」。改動機制本身時，要
特別去想「這個判斷背後偷偷假設了什麼場景/出生點配置」，不能只看新需求本身合不合理。

### 後續兩個追加請求：徹底拿掉緩衝間隙 + 平台邊緣摔落
用戶重申「太高狗爬不過去都要改成語意標籤標記牆」，把 Bug 4 留的 1 階間隙也拿掉，兩段樓梯
最高一階外露面補丁改成涵蓋完整外露高度（`HALF_Z`/`TOP_Z`，不再減去任何 `STEP_H`）。安全性
確認：跨過去那一刻機器狗實際高度（landing/Floor_2 上約 0.87-0.90m / 1.46-1.49m）本來就高於
補丁範圍（含 0.15m Z-margin 後上限 0.75m / 1.35m），所以補丁再怎麼加高都不會蓋到機器狗真正
站上平台那一刻的高度，不會重演 open_scene_stairs.py 那次「補太滿導致在頂端反覆震盪」的舊坑。

改完後用戶接著回報：機器狗成功爬上平台，但**從平台邊緣踩空掉下去**。平台（landing）四邊
只有兩邊跟 Stair A 出口 / Stair B 入口相接，另外兩邊（遠離 Stair A 的 +X 邊、遠離 Stair B
入口的 -Y 邊）完全沒有東西接著，踩下去就是直落 ~0.5m 到一樓地板。這兩邊原本因為「樓梯不可以
有牆壁」的要求被拿掉了實體護欄（見上面主要條目），但沒有補上語意標記，導致地板類的
`SEG_ID_FLOOR` 一路延伸到邊緣，機器狗的感知完全看不出邊界在哪。

修法：跟樓梯外露面/側面同一套邏輯——語意專用（`_tag_wall_region`，無碰撞）WALL 補丁，貼在
平台這兩個懸空邊。**這裡踩了一個新坑**：一開始把補丁高度設成只涵蓋「平台下方到地面的落差」
（0 到 `HALF_Z`），結果完全沒用——因為 `viplanner_control.py` 的 wall-proximity 檢查是拿
「機器狗自己當下的 z」去比對補丁的 Z 範圍（±0.15m margin），機器狗真正走在平台上時的高度是
~0.87-0.90m（平台面 HALF_Z=0.60 加自己站立的體態偏移 ~0.28m），遠高於補丁只到 0.60+0.15=0.75
的範圍，兩者完全對不上，等於補丁形同虛設。改成補丁高度用 `HALF_Z + 0.4`（=1.0m），確保機器狗
真正站在平台上巡邏時的高度落在補丁的匹配範圍內，這樣避牆機制才會真的生效。

修好後驗證：機器狗爬上平台後 z 穩定停在 0.85-0.90m（沒有再摔回地面），即使在平台上繞了
~2000 步找 Stair B 入口也沒有意外掉落，最後第 4756 步順利到達二樓目標。

**教訓**：語意補丁的「高度覆蓋範圍」不能只從幾何直覺（「這個空間有多高」）去設，要對照
`viplanner_control.py` 實際拿什麼值去比對（這裡是機器狗自己當下的 z，不是補丁底下空出來的
落差空間），否則補丁看起來合理，實際上永遠不會被觸發。


## 2026-10-01 — Stair.usdc 場景接入 + 欄杆語意

### Flight4 被當成「下一階」把狗拉回頭
一開始 Flight3 沒碰撞（只到 2F），但 Flight4 有標 STAIRS。狗爬到 Flight2 頂端、還沒踩上 2F 地板
（`on_floor_reaching_goal` 仍 False）時，「選下一個可爬的樓梯 footprint」挑到了跟 Flight2 XY 重疊的 Flight4，
planning_goal 被拉回身後，狗在樓梯井欄杆邊打轉。**原則：走不到的樓梯不能標 STAIRS**；另外目標點離任何
STAIRS footprint < 1.5m（`_ENTRANCE_SEEK_MARGIN`）會被當成樓梯口目標。

### 欄杆端柱卡死
平台目標點放在平台正中 (2.2, y) 時，狗爬完 Flight1 會斜切過去，撞上欄杆末端柱子 (2.2, 3.42) 卡 5000+ 步。
改成目標點在樓梯正前方 (3.4, 4.49)，讓狗先直走上平台再迴轉，迴轉路徑離柱子夠遠。

### 踏面數算錯（13 階高 ≠ 13 踏面）
每段樓梯 13 個 riser，但最後一個 riser 直接上平台，所以只有 12 個踏面（3.24m = 12×0.27）。一開始用 13 個
踏面標 STAIRS，Flight1 的標註伸進平台 0.27m、Flight2/4 整段錯位一格。從 mesh 頂點 z/y 的 unique 值才看出來。

### 欄杆 → ViPlanner 的 fence 類別
ViPlanner 原本的 `VIPLANNER_SEM_META` 就有 `fence`（OBSTACLE_LOSS），沿用它的顏色就不需要改網路輸入；
cost map 端在 `build_costmap.py` 跟 wall 一起把 loss 拉到 20。

### 環境注意
- 主機原生 `isaacsim-5.1.0` 因 glibc 太舊跑不起來，一律用 docker。
- 在 docker 裡用 `python.sh` 跑 pxr 腳本：檔名不能叫 `inspect.py`（跟 stdlib 撞名導致 SimulationApp 循環 import）；
  `SimulationApp` 會吃掉 stdout，要把 `sys.stdout` 導到檔案。
- 啟動偶發 Hydra `HdRenderIndex::_InsertSprim` segfault（exit 139），跟腳本無關，重跑即可。

### 疊層樓梯間（Flight1 正上方是 Flight3）讓「挑下一階」邏輯出錯（2026-10-01）
`viplanner_control.py` 的 flight 分群、入口挑選原本只看 XY 相鄰與「高度爬得上去（上限）」，在 XY 完全重疊的
疊層樓梯間會：(1) 從 Flight3 頂挑到隔著欄杆的 Flight4 第一階 → 撞欄杆；(2) 只改 `_flood_flight` 加高度連續後，
入口分群仍只看 XY，入口被選成正下方樓層的 Flight2 第一階，真正要爬的 Flight4 被「太高樓梯要繞開」nudge 推開
→ 入口前來回 12000 步。**教訓：同一個「flight」概念在檔案裡有好幾份實作（_flood_flight、入口分群、
_cluster_stair_flights、nudge 的 target flight），改一處要全部一致**；候選台階要有上下限（不能是正下方樓層的）。

## 2026-10-02 — 深度語意除錯紀錄（重要結論）
- **Isaac Camera 沒設 clipping range → 近端裁切 1 m**：1 m 內的幾何被切掉變透明，射線穿過樓板頂面打到樓板底面，
  深度看起來比地板低 0.2 m（樓板厚度）→ 平地出現假台階。用「固定相機拍水平地板」的控制實驗定位（深度 min 剛好 1.0）。
  深度確實是 distance_to_image_plane（垂直距離），中間一度誤判成直線距離，已更正。
- 用收集資料反推相機參數（HFOV、pitch、fy）都擬合不出來，因為混了多種誤差；**控制實驗**才是快速定位的方法。
- **AABB 標準答案在疊層樓梯間有先天錯誤**：上方樓梯底面落在其樓梯盒子內 → 被標成樓梯；樓板側面落在地板盒子內 → 標成地板；
  欄杆盒子包到踏面邊緣。評估時排除頭頂以上像素、排除欄杆。
- 相機在頭後面（0.15,0,0.15）時被自己的頭擋；移到 URDF front_camera 後，0.2~0.35 m 的深度是真實台階，min_range 要 0.21 不是 0.35。
- 踢腳板頂端是 17 cm 高的窄水平條 → 被當成一階 → 修法「窄條且緊貼高牆才排除」+「立面只歸屬樓梯」。
  只用窄條條件會把斜看的細長踏面也濾掉（樓梯辨識 90%→44%）。
- mesh 點雲：法向量正負號可分頂面/底面（Stair.usdc winding 一致）；地墊/色帶是零厚度薄片，其底面不能當天花板；
  實心樓梯底下的地板被蓋住 → 不可輸出成障礙物（2D cost map 會把整段樓梯變高成本）。


## 2026-10-06 — combine_without_BlendersWall：Gaussian splat 在 Isaac Sim 5.1 不顯示 → 轉 NuRec
- 場景 `Lab_scene_v1.usdc`：可見層只有 `OfficeSplat`（payload `3DGUT/office_bounded_room_zup.usdz`，
  `ParticleField3DGaussianSplat`，OpenUSD 26 的 schema，76 萬顆、SH 3 階）；`PhysicsLayer` 碰撞方塊在原檔就是 invisible，
  而且沒有顏色（displayColor 欄位存在但沒有值）。
- Isaac Sim 5.1 只認 NuRec（`UsdVol.Volume` + `OmniNuRecFieldAsset` + `.nurec` msgpack/gzip）。3DGRUT README 寫明
  NuRec 支援 Kit 107.3–110.1 / Isaac Sim 5.0–6.0；ParticleField 要 Isaac Sim 6.0。
- 轉檔：`isaac-sim-docker/home/tools/splat_to_nurec.py`，用 Isaac 的 python（`docker exec isaac-sim /isaac-sim/python.sh
  splat_to_nurec.py in.usdz out.usdz`）跑官方 `threedgrut.export.scripts.transcode --format nurec`。
  - Isaac 的 pxr 沒有 ParticleField schema class → monkeypatch importer，改用 `prim.GetAttribute()` 直接讀屬性
    （四元數照官方 importer 轉成 [w,x,y,z]）。post→pre activation 的轉換交給官方 adapter。
  - 3DGRUT 的 import 鏈會拉進訓練用套件（ncore、datasets、simplejpeg、tensorboard…）→ 用 meta_path finder 換成空殼模組。
    額外套件裝在 `tools/pylibs`（msgpack、plyfile、omegaconf、rich、tqdm、einops、kornia）。**不要讓 pip 把 numpy 裝進
    pylibs**（曾裝到 numpy 2.4，已刪除，會跟 Isaac 的 torch 衝突）。
  - 座標：官方輸出把原本 usdz 內的 local-to-world（含 roomAlignment）寫在 volume 的 `xformOp:transform:sourcePose`，
    payload 到 OfficeSplat 後位置跟原本一致（截圖確認狗站在 splat 地板上）。
  - NuRec 要求的 rtx render settings 存在它自己 layer 的 customLayerData，只有當 root layer 才生效 → 場景腳本用 carb 設定。
- 深度：NuRec splat 只出現在畫面，`distance_to_image_plane` 深度看不到 → splat 模式 ViPlanner fear≈1.0。
  `SHOW_PHYSICS=1`（顯示碰撞方塊）時 fear≈0.1、650 步到達。
- 解法（用戶選擇「畫面 splat＋深度用方塊算」）：`depth_from_aabbs()` 對 75 個 AABB 做 slab 光線求交，投影模型跟
  `reproject_depth_to_world` 完全相反（來回測試誤差 < 0.01mm）；命中距離 +1mm，避免點剛好落在面上時被
  `classify_world_points` 的邊界判斷因浮點誤差漏掉（不加的話約 6600/36864 像素沒有類別）。
  修改前的 `viplanner_control.py` 備份在 session scratchpad（`viplanner_control.py.before_proxy_depth`，暫存）。
- 偶發：關掉 container 後馬上重開，Isaac Sim 常在開場景前崩潰（`pure virtual method called`），多等約 10 秒再開就正常。

## 2026-10-13 — 換 leo 的踩坑紀錄
- 用戶隨身碟複製的大檔損毀：`base.STL`（54 MB，189 萬個頂點是亂碼）、權重一度 0 bytes。機身改用 HIMLoco 內同一份（md5 不同但大小相同、
  URDF 完全一致）。未動用戶 `1013/` 內檔案，轉檔來源另組在 `robots/leo/src/`。
- `fire_dog/leo.usd` 是另一套定義（LF_HAA 命名、軸向/小腿零點/機身前後都不同），不能直接配 HIMLoco 權重；一律用訓練同一份 URDF 轉 USD。
- 機身 108 萬面：PhysX 啟動時現算凸包會卡死在 timeline.play() 後（>7 分鐘）。離線先算凸包（492 面）當碰撞，外觀仍用原檔。
- **URDF 匯入器預設 drive type = acceleration**：力矩 = 慣量×kp×誤差，70 kg 的 leo 只出 ~40 Nm 直接趴下。轉檔後改成 force。
- Articulation 用 `joint_names=` 會把固定關節也算進編號（leo USD 保留腳掌等固定關節）→ index 12 越界。改用從 dof_names 算的 `joint_indices`。
- 無 inertial 的 link（base、imu_link）PhysX 會各補 1 kg → 設 0.01 kg。出生高度不能用零姿勢 bbox（腿伸直 0.82 m），leo 直接放 0.60 m。

## 2026-10-14 — 從 progress.md 封存的舊段落（09-16 ~ 10-06 原文）

### 2026-09-16 — 自建樓梯場景 + v3 模型 + 入口轉彎修正（本次）
- 場景/控制程式碼已搬到本專案內：`isaac_sim_bridge/viplanner_control.py`（控制迴圈）+
  `/home/itri/isaac-sim-docker/home/open_scene_stairs.py`（場景建置，掛載進容器，不在本專案內）。
  場景是自建的兩段式樓梯（Stair A → landing → Stair B → 2F 房間），不是 09-10 那個一般房間場景。
- 修過的重大 bug（細節見 history.md）：
  - 樓梯背面「露出的最後一階正面」被誤判成可走地板 → 用 `_tag_wall_region`（純語意標記，
    無實體碰撞）補上牆標記，且刻意在頂部留 ~0.3m 的 Z 間隙避免又卡在每段樓梯頂端來回震盪。
  - `wp_idx` 判斷 Stair A 該不該重標成牆的門檻，跟 5-waypoint 版本對不起來。
  - `_nearest_wall` 原本只看 XY，會把 2F 房間門框誤判成就在 Stair A 入口正上方 3 公尺處的近牆
    → 加了 `_WALL_Z_MARGIN`（目前 0.15）。
  - 機器狗在樓梯入口「不敢轉彎進去、繞一大圈」→ 根本原因不是牆壁安全邊界，是入口 waypoint
    幾乎與地面同高（z_gap 太小），完全沒有觸發樓梯導引，整段路交給 ViPlanner 自己的原始預測，
    而它就是不想直線靠近樓梯。已修正（見 history.md 完整除錯過程）。
- v3 微調模型：`training_data/models/plannernet_envstairs_scene_ep30_..._stairs_finetune_v3`
  （實際訓練了 60 epoch，目錄名字沒改到，val_loss=10.14，已確認收斂）。v2 整條訓練資料
  （點雲/cost map/rollout）都是修好「樓梯背面」bug 之前收集的，等於沒學到正確語意，
  v3 是用修好之後重新收集的資料全部重跑一次（build_cloud → build_costmap → 收集 rollout
  → 微調）。v2 模型跟最原始的 base checkpoint 都保留著沒刪。
- 目前用的即時互動參數（`viplanner_control.py`）：`REPLAN_EVERY_N_STEPS=8`、
  `COMMAND_HOLD_STEPS=8`、`HEADING_SMOOTH_RATE=0.4`。
- 又修了兩個相關 bug（同一天，入口修正之後才發現）：
  - 樓梯護欄原本高度跟著每一階「局部」升高（鋸齒狀輪廓），機器狗爬到比較高的階梯往後看時，
    視線會直接越過矮的護欄段看到走廊外的地板 → 改成整段樓梯護欄統一用最高高度（不再鋸齒狀）。
  - 機器狗爬完 Stair A 站上 landing 後常常卡住不動、常常正對著牆 2000+ 步才繼續 → 跟入口那個
    bug 是同一種模式：`near_stairs_entrance` 的判斷也錯誤地被 `on_stairs` 這個會抖動的布林值
    擋住，站在平台上時 XY 偶爾跟樓梯的判定框重疊，讓幾何導引被誤關掉。已拿掉這個門檻。
  - 兩個修正都是用「在控制迴圈裡存下語意圖片、直接用眼睛看」抓到的，不是純靠猜參數。
- 實測驗證：入口 + landing 修正後，連續跑了 20 個 episode，**全部順利通過入口、平台、爬完
  整段樓梯到 waypoint 4/5**，成功率 100%，沒有再卡住或貼牆。這是目前為止最穩定的一次驗證批次。

### 2026-09-17 — 護欄安全邊界加回對角線
- 之前為了解決入口轉彎的問題，把樓梯護欄（guardrail）的軟性避牆邏輯直接整個跳過
  （`effective_wall_dist=0.0`），只靠 PhysX 實體碰撞擋。現在確認真正的入口/landing 根因
  （見上面 09-16 那兩個 bug）已經修好，這個跳過就不再需要，而且會讓機器狗貼著護欄蹭過去
  （沒有任何提前的軟性推開）。已改回跟真牆一樣的保守半對角線邊界（`GO2_HALF_DIAGONAL`）。
- 實測：改回後連續 8 個 episode 都順利通過，沒有回歸。

### 2026-09-17 — 暫時拆掉樓梯/平台的牆做觀察，補了兩個安全機制
- 用戶要求暫時拆掉樓梯兩側護欄跟平台四周的牆（`open_scene_stairs.py` 新增
  `DISABLE_STAIR_SIDE_WALLS` 開關，目前設 `True`，要恢復正式跑就改回 `False`）純粹是為了
  觀察機器狗在沒有實體牆擋著時的原始行為。
- 拆牆後發現機器狗會走出樓梯側邊掉下去，因此補了兩個機制（純控制邏輯，不影響場景本身/
  語意圖片）：
  1. **高度可走性檢查**：選擇要爬的下一個樓梯 footprint 時，不再單純選「XY 最近的」，優先選
     「從目前高度爬得上去的」（高度差在 0.35m 內），避免瞄準語意標記是樓梯、但實際高度差太大
     爬不上去的目標。
  2. **虛擬邊界牆**：重用原本護欄用的同一套安全邊界機制（半對角線 + 逐步推離 + 嚴重時強制
     轉向），但不建實體物件——直接從樓梯階梯本身的真實寬度算出左右邊界，餵進控制迴圈的
     `wall_boxes_xy`（不進 `aabbs`，所以機器狗自己看到的語意圖片不會多出假牆）。兩段樓梯各
     自算，不寫死寬度常數。
- 實測：拆牆後連續 2 個 episode 都順利爬完整段樓梯、通過平台，沒有再掉下去。

### 2026-09-17 — 往「實機也可以用」的方向推進，決定先用深度分類器這條路
- 用戶明確要求這套系統之後也要能上真實硬體，不能只在模擬裡跑。現在的架構有個根本問題：
  控制迴圈幾乎所有安全機制（牆壁邊界、樓梯導引、虛擬邊界牆）都是直接查場景腳本自己記錄的
  真實座標（`aabbs`），真實機器人沒有這種上帝視角的標準答案可查，只有深度相機 + 語意分割。
- 查了 `viplanner_core`（ViPlanner 原始庫）本身，發現它已經內建一個為真實部署設計的方案：
  `viplanner/utils/geometric_semantic.py` 的 `compute_geometric_semantic()`——純靠深度算高度，
  不用場景標籤、不用訓練模型，天生就能上真機。缺點是只有「地板／障礙物」兩類，沒有樓梯專屬
  類別（樓梯在它眼裡是一條條「可走／不可走」交錯的高度條紋，不是一整塊）。
- **決策（用戶明確拍板，記錄備查）**：先走這條「深度分類器」路線——把 `viplanner_control.py`
  的語意輸入換成 `compute_geometric_semantic()`（已完成，`VIPLANNER_USE_GEOMETRIC_SEMANTIC=1`
  開關），收集新的訓練資料（`training_data/stairs_scene_geometric/`，拆牆狀態下收集，因為
  真機也不會有護欄），從最原始的 base checkpoint（不是 v3）重新微調出一版新模型。
  **如果這樣效果不夠好，再考慮訓練一個真正的語意分割模型**（用 RGB／深度辨認「這是樓梯」的
  形狀，而不是只看高度）——這是更大的工程（要自己訓練一個分割網路，還要處理模擬訓練、真實
  相機測試之間的 sim-to-real 落差），先不做，除非深度分類器這條路驗證不夠用。
- 目前進度：資料收集中（30 episode，拆牆狀態），收集完後接著微調。

### 2026-09-22 — 新的單轉角 L 型樓梯場景，接上狗 + 整套 ViPlanner，一連串除錯後單一目標點模式穩定跑通
- 新增 `l_shaped_stair_scene_with_walls_viplanner.py`（專案根目錄）：把使用者自建的
  `l_shaped_stair_scene_with_walls.py` 場景改用 `UsdGeom.Cube + UsdPhysics.CollisionAPI` 重建、逐箱標
  `SEG_ID_FLOOR/STAIRS/WALL`，加狗、接上既有的 `isaac_sim_bridge/viplanner_control.py::run_walk()`（v3
  模型 + HIMLoco checkpoint）。單轉角樓梯：1.5m 寬、N=6 階、landing 1.5x1.5m、TOP_Z=1.20m。
  `aabbs.json` 寫到獨立的 `training_data/l_shaped_stairs_scene/`，不影響主線場景自己的訓練資料。
- 手動組 docker run 指令時記得帶上 `isaac-sim-docker/home:/home/isaacsim:rw`（HOME 掛載，缺少會導致
  texture cache 寫入失敗、整個啟動失敗）、`fire_dog:ro`、`viplanner_ckpt:ro`、`training_data:rw`。
- 接上後陸續修了 7 個問題（樓梯側面摔落、入口震盪、RViz 側面誤標綠色、最高一階背面補丁間隙太寬後來
  乾脆完全拿掉、單一目標點模式下的三個新 bug、平台懸空邊緣踩空摔落），完整除錯過程見 `history.md`
  同日條目（含一個補丁高度設錯導致形同虛設的教訓：要對照 `viplanner_control.py` 實際拿什麼值比對）。
- **目前狀態**：4-waypoint 版本與單一目標點版本（`L_SHAPED_SINGLE_GOAL`，現已改成永久只用二樓目標點
  這一個 waypoint，出生點也改到二樓目標點正下方 z=0）都已驗證穩定成功，z 軌跡全程單調爬升無摔落。
- 語意標記原則（用戶確認）：太高（不可一步跨上去）一律標記牆；有高度差但可爬的才是樓梯；其餘是地板。
  目前場景的標法已符合這個原則（含兩段樓梯的最高一階外露面、左右側面補丁）。

### 2026-10-01 — 使用者提供的 Stair.usdc 場景接上狗 + ViPlanner、新增「欄杆」語意、重新訓練 v5（進行中）
- 場景：`~/Downloads/Stair.usdc`（折返樓梯間 1F→Flight1→平台→Flight2→2F→Flight3→平台2→Flight4→3F，
  階高 0.18m／踏面 0.27m，13 階高、12 個踏面）。原檔的 Flight3、Landing2 只有外觀沒有碰撞，經用戶同意
  **直接寫進 Stair.usdc**（複製 Flight2/Landing 的碰撞設定，原檔備份為 `Stair.usdc.bak`）。
- 入口腳本 `/home/itri/isaac-sim-docker/home/open_scene_stair_usd.py`；啟動 `isaac-sim-docker/run_stair_usd.sh`
  （`COLLECT=1 EPISODES=30` 為收集模式）。場景照原檔載入，不另外加燈／牆／碰撞（用戶明確要求）。
- 目標點：平台 (3.4,4.49) → 2F 中間 (2.2,-1.32) → 平台2 → **3F 中間 (2.2,-1.0,z 9.71)**；`max_steps=26000`。
- **新語意 RAILING（欄杆）**：`SEG_ID_RAILING=4` → ViPlanner 原有的 `fence` 顏色 [76,0,153]，cost 跟 wall
  一樣 20；控制迴圈的牆壁安全邊界也包含欄杆。改了 `camera_semantics_lab.py`、`viplanner_control.py`、
  `build_cloud.py`、`build_costmap.py`（`ROOT` 可用 argv[1] 指定場景）。
- **控制邏輯修正（`viplanner_control.py`，疊層樓梯間）**：選下一階/入口時加「視線被牆或欄杆擋住就不選」
  (`_los_blocked`)；flight 分群（`_flood_flight` 與入口分群）加高度連續條件；入口/下一段候選排除低於狗 1m 以上
  的台階；「太高樓梯要繞開」略過頭頂上方的樓梯。修改前原檔：scratchpad `viplanner_control.py.orig`（暫存）。
- 2026-10-01 追加（狗從樓梯井縫掉到 1F，再從 1F 無牆的 y=-2 邊緣掉出場景）：`viplanner_control.py` 加
  (a) 入口直線碰到牆/欄杆時，先走到入口前 0.8m 的準備點 (`_entrance_approach`)；(b) base z < -2m 直接結束 episode。
  欄杆語意 AABB 前後各加 5cm。**這兩項在 30 episode 收集中途才改，本批資料是用舊控制收的**（資料本身有效）；
  墜落中的樣本由 `training_data/filter_fallen_samples.py` 在訓練前自動刪除（串接腳本已加這步）。
- 驗證：修正後收集的前 3 個 episode **3/3 走到 3F**（6849~12208 步）。修正前會卡在 Flight3 頂欄杆、
  Flight3/4 入口前來回。v3 爬 0.18m 階時偶爾會中途轉身（fear 偏高），靠重新訓練改善。
- 訓練（用戶選擇）：只用此場景資料、從 v3 微調、收集 30 episode → `train_stair_usd.py`
  → `models/..._stair_usd_railing_v5`。串接腳本 `training_data/collect_then_train_stair_usd.sh` 收集完自動訓練，
  log 在 `training_data/logs/`。**若沒收斂：補收到 50 個 episode 再重訓**（用戶指示）。

- **資料與模型（2026-10-01）**：第 1 批 `stair_usd_scene` 19 ep（舊控制）、第 2 批 `stair_usd_scene_b` 11 ep（新控制）
  → v5（從 v3 微調 30 epoch，best val 21.72 @ epoch 26，最後 3 epoch 無進步，接近但未完全收斂）。
  用戶決定補到 50 ep：第 3 批 `stair_usd_scene_c` 20 ep（含跌倒偵測）→ 三批一起從 v3 重訓 **v6**（進行中）。
  串接腳本 `collect_then_train_stair_usd.sh` 會在 log 出現「DATA COLLECTION complete」時自動關 container
  （Isaac Sim 收完資料不會自己關）。`run_walk` 另加：機身傾斜 >70° 持續 200 步 → FELL OVER 結束 episode。
- 控制迴圈是跟著 Kit 畫面更新跑（約 60Hz），不是訓練時的 50Hz（dt 0.005 × decimation 4），步態在 0.18m
  階梯偶爾跌倒可能與此有關。

### 2026-10-02 — 往實機推進：深度語意（階段 1）完成、無標註 cost map（階段 2）進行中
- 用戶目標：**要能上真實機器狗**。分四階段：①深度語意 ②無標註訓練流程 ③控制不依賴 aabbs（局部高度圖）④實機（定位、相機規格、雜訊）。
- **相機修正（影響所有之後的資料/模型）**：`CAMERA_NEAR_CLIP=0.2`（原本沒設 → Isaac 預設 1 m 近端裁切，1 m 內被切成透明，
  會看穿樓板）；相機移到 URDF `front_camera` (0.327, 0, 0.043)、仍往下 8.6°（原本在頭後面會被頭擋）。v3~v6_cont 都是舊相機。
- **`geometric_semantic_v2.py`**（isaac_sim_bridge）：純深度 → 地板/樓梯/障礙物。重力轉正 + 每幀地面平面校正、法向量分水平/垂直、
  以狗為中心 5 cm 高度格、從腳底層 flood fill（高度差 ≤0.22）、短距離台階邊緣 + 擴散整個踏面、窄邊（踢腳板）過濾、
  立面只歸屬樓梯。乾淨評估資料 `stair_usd_eval_clip02_v2`：整體 88%，地板 69%，樓梯 90%，障礙物被判可走 4.4%（欄杆不計分，
  用戶決定欄杆就是細柱子）。評估腳本 `eval_geometric_semantic_v2.py`。
- 控制程式開關 `VIPLANNER_GEOMETRIC_SEMANTIC_V2=1`（`run_stair_usd.sh` 用 `GEO_V2=1`）；腳底高度用關節 FK。導引/防撞仍查 aabbs（階段 3 才改）。
- **修 bug**：`run_walk` 的「封鎖 Stair A」用 y 中心挑台階，在 Stair.usdc 把 Flight4 最上一階在最後一段改標成牆 → 新參數
  `block_first_flight_after_wp`（預設 3 不變，Stair.usdc 傳 None）。v5/v6 訓練資料最後一段有此假牆。
- **無標註 cost map**：`training_data/tools/export_collision_mesh.py`（docker 匯出碰撞三角形）+ `build_cloud_from_mesh.py`
  （朝上面才可站、上方 0.3 m 需淨空、從出生點 flood fill、藏在 ≤2.1 m 可走面下方的東西丟掉）。
- v6_cont（v6 再 30 epoch）best val 20.59，已接近收斂，但資料有假牆/舊相機問題。
- **進行中**：`stair_usd_geo` 收 50 episode（GEO_V2=1、新相機）→ 自動從 v3 訓練 `stair_usd_geo_v7`（`logs/chain_stair_usd_geo_v7.log`）。

### 2026-10-06 — 開啟新場景 combine_without_BlendersWall
- `combine_without_BlendersWall 1.zip` 解壓到 `scenes/combine_without_BlendersWall/`（`Lab_scene_v1.usdc` + `3DGUT/office_bounded_room_zup.usdz`）。
- 啟動：`isaac-sim-docker/run_combine.sh` → `home/open_scene_combine.py`（只開場景，無狗/ViPlanner）。已成功開啟（246 prims）。
- 已接上 Go2 + ViPlanner（原始 base 模型，T7 外接碟沒掛載 → `run_combine.sh` 改掛 `base_checkpoint/`）。
  出生 (1.2,-3.8) → 目標 (-0.9,3.0)，episode 1 在 2670 步到達。語意 AABB 由碰撞方塊產生（地板/其他全部是牆）。
- **畫面全黑的原因**：唯一可見的是 `ParticleField3DGaussianSplat`（新版 OpenUSD 的 splat schema），Isaac Sim 5.1
  的 log 沒有任何 splat 相關訊息，畫面也沒畫出來；碰撞層又設成隱形 → 畫面上幾乎沒東西。深度相機一樣看不到 splat →
  ViPlanner 等於沒看見任何東西（fear 一直 0.99），能到達目標是靠 aabbs 防撞。替代方案：`SHOW_PHYSICS=1` 改看碰撞方塊。
- `SHOW_PHYSICS=1` 重跑（碰撞方塊沒有材質 → 腳本在記憶體裡依群組上 displayColor）：fear 降到 ~0.1、速度 0.5~0.6，
  episode 1/2 分別 650/675 步到達（splat 模式要 2670 步）→ 之後這個場景用 SHOW_PHYSICS=1 跑。
- **splat 顯示出來了（2026-10-06）**：Isaac Sim 5.1 只支援 NuRec 格式（`omni_nurec_types`），用官方 3DGRUT `transcode`
  把 ParticleField 轉成 NuRec → `3DGUT/office_bounded_room_zup_nurec.usdz`（原檔保留），`open_scene_combine.py` 把
  OfficeSplat 的 payload 換成這個。轉檔工具：`isaac-sim-docker/home/tools/splat_to_nurec.py`（細節見 history.md）。
- 深度相機看不到 NuRec splat → 新增 `VIPLANNER_PROXY_DEPTH=1`（`viplanner_control.py::_get_depth` +
  `camera_semantics_lab.depth_from_aabbs`）：畫面顯示 splat，ViPlanner 深度用碰撞方塊光線求交算。預設關閉，其他場景不受影響；
  `run_combine.sh` 在 splat 模式預設開啟。實測 3/3 episode 到達（636/653/653 步，fear 0.05~0.2）。
  代價：幀率從 ~47 掉到 ~27 fps（raycast 在 CPU 上每次約 73ms），有需要可改成 GPU/torch。
- 起終點（用戶要求拉遠、再換一組、不要在桌子下）：現在 出生 (3.4,4.0) yaw -135° → 目標 (1.0,-3.5)，路徑約 11 m。
  之前用過 (1.2,-3.8)→(-0.9,3.0)、(-2.5,-3.4)→(3.4,3.8)。
- **桌子當實心**（用戶要求不要在桌子下）：`_build_aabbs` 把 `/World/PhysicsLayer/Table/*` 每張桌子合成一個從地板到桌面的
  WALL 方塊（語意＋proxy 深度都看成實心，物理碰撞不變）。實測 2 個 episode 軌跡都沒進到任何桌子 0.3m 內，ep1 2397 步到達。
- 小問題：DomeLight 貼圖路徑寫死 `./lab_alignment_v4/textures/color_0C0C0C.exr`（zip 內實際在 `3DGUT/textures/`），只影響環境光背景。


## 2026-10-14 — 10-13/14 的踩坑紀錄
- 串接腳本 `collect_then_train_stair_usd.sh`：container 還沒起來就掛上去會立刻誤判「收集結束」而去訓練（FileNotFoundError）；停收集前要先殺串接腳本，否則它會拿不完整資料訓練。
- 在同一行指令裡用 `pkill -f`/`pgrep -f` 找某腳本名，會比對到自己那行 shell 而把自己殺掉（exit 144）；把邏輯寫進獨立腳本、用 PID 殺。
- growin 開不起來：腿部碰撞網格 6~10 萬面，PhysX 在 timeline.play() 後 cooking 卡死。用 scipy ConvexHull 事先算 `*_hull.STL`，URDF 副本的 collision 指過去再轉 USD（跟 leo 機身同一招）。
- Isaac Gym 依名稱排序同層關節 → policy 的關節順序固定是 FL, FR, RL, RR，跟 URDF 內順序無關。
- 關掉 container 後馬上重開的啟動崩潰又出現一次（10-14，停掉後約 10 秒重開，當在 Hydra `HdxTaskController` 建立時），重開一次就正常。

## 2026-10-14 — GUIDANCE=perception 下樓除錯紀錄（stair_perception.py 每條規則的由來）
- 相機看不到腳邊台階：在樓梯上時可見的最低樓梯點常在腳上方 0.5~0.7 m → REACH_FROM_FEET 0.45 → 1.0。
- 爬到頂前 3 階就切成「同一層」：目標點浮在地板上 0.35 m → 加「機身水平一段時間」才切換（狗斜切撞中間牆、腳卡住）。
- 機身一水平就原地轉找下一段 → 後腳還在台階上就轉而卡住 → 不在台階上原地轉（blind 直走）、改用「水平後走出 0.9 m」判斷走完（時間判斷在下樓快速時會衝出 2.5 m）。
- 平台上一直水平 → 走向下一段途中一轉身就丟掉偵測 → 水平規則只在「已經上過這段樓梯」後才生效（on_flight）。
- 撞牆時機身仰起被當成「還在台階上」直走 → 只算順方向的傾斜（下樓低頭、上樓抬頭）。
- 坡度 0.25 會收進遠處半看見的樓梯（0.3~0.4）把狗帶走 → MIN_SLOPE 0.45。
- 側/後相機在樓梯上就看到下一段 → 中途改目標、在台階上掉頭 → 還在這段的台階上時忽略不同方向的偵測。
- 全速下樓一次掉 3~4 階、前傾 35° 步態卡死 → 下樓限速 0.3 m/s。
- 1F–2F 平台：剛下來那段和下一段同向相鄰 → 走回頭路 → 記住走過的樓梯（方向 + 橫向位置 + 高度範圍；高度必要，因為折返樓梯每隔一段同向疊在正上方）；連通判斷高度差 0.45 → 0.25 避免兩段樓梯被併成一段。
- 「走完」只在看不到樓梯時檢查 → 還看得到最後幾階就一直往前撞牆 → 先檢查走完再收偵測。
- 結論：逐幀判斷 + 一堆規則在平台這種多段樓梯同時可見的地方很脆弱；下一步建議改成持久的樓梯地圖。

## 2026-10-14 — 樓梯地圖上樓修正、腿部里程計重寫

### 上樓時一直往下走（stair_map.py）
- 原因：折返梯兩段在平台相接，某幀兩段的點被分成同一組，擬合出歪掉的斜面（方向 (-0.3,-0.9)），最低點被拉到 1.7 m（平台約 2.4 m）。
  沿這條歪中線走時實際踩在剛爬完的第一段上 → 一路往下。
- 修正：`_fit_flight` 內點比例 <60% 丟掉（MIN_INLIER_FRAC）；`_match` 加平面高度比對（存 `z_c`，平面 z = slope*s + z_c）；
  walk 階段腳高往反方向超過 0.3 m（WRONG_WAY_DROP）就把該段標成走過、換一段。
- 接著發現：2F→3F 那段被拆成兩段（中心線差 1.5 m，一條在牆裡）→ `_merge()` 合併同向、同平面、側向相鄰的兩段（正在 walk 的那段不合併，否則雜片把終點拉遠、永遠走不完）。
- 再發現：第二段最低點仍被雜點拉低，與腳高差 0.7 m > 0.6 容許 → 平台上找 1600 步。改成入口 = 平面上與「選定時的腳高」等高處（`frame(..., entry_z)`），候選要求再往上/下 ≥1.5 m（MIN_RISE）。
- 再發現：狗在第一階偏中線 0.9 m 貼隔牆卡住 → 開始 walk 要離中線 0.4 m 內（LINE_TOL）；入口附近 50 次沒前進就退回 approach 重新對正（WALK_STALL_TICKS）。
- 結果：上樓 3/3 到 3F（old_logs/launch_growin_up_run5_3of3.log）。離線驗證腳本 scratchpad/map_test.py（用 growin_rgb_test 錄影）。

### 腿部里程計
- 舊的 `go2_foot_positions_body_frame` 用 Go2 尺寸與軸向，且用機身 z 而非重力方向 → growin 上真爬 +8.6 m 估成 −10.6 m。
- 新 `leg_odometry.py`：URDF 讀關節鏈（growin 大腿 0.35、小腿 0.30，各腳軸向正負不同）；腳相對機身高度 h = -(p·g_body)，g_body 來自 IMU。
  用真值驗證 h 與模擬世界座標相符（誤差 5e-8 m）→ 幾何正確，誤差全來自「哪隻腳算支撐」。
- 接觸來源比較（錄檔 up1_ep1，整趟真爬 +8.60 m，積分支撐腳的 Δh）：
  模擬 ContactSensor −21.3 m（12% 時間四腳全 0，且起腳時仍報接觸）；PD 指令力矩算腳力 >60 N −4.7、>150 N −1.46、力加權(>100) −1.40；
  「最低兩腳」−11.3；加「腳不在上升」條件反而高估 +2~4 m。誤差集中在樓梯（−1.64 m / 8.5 m = 19%），平地 +0.28 m。
  原因：樓梯上推蹬腳在受力時已經上升。模擬回報的量測力矩（get_measured_joint_efforts）總和平均 588 N 較合理但平地就漂 +1.5 m/600 步。
- IMU(真實加速度+雜訊)+腿速度的垂直卡爾曼濾波初版（scratchpad/legodom_kf.py）多數發散，腿速度量測雜訊設太小、門檻待調。

### 腿部里程計（10-15 續）— 濾波器、500 Hz 物理、出生高度
- 用戶選「IMU + 腿的濾波器」（要部署實機）。過程中發現的根因，依序：
  1. 我 10-14 說「FK 與真值誤差 5e-8」是自己跟自己比（兩邊都用 FK），不算驗證。後來錄模擬腳 prim 世界座標（XFormPrim）比對：四腳相對位置誤差 0，FK 才真正確認正確。
  2. 「腳一直滑」：腳是半徑 ~6 cm 的球，站立時小腿轉動讓球心水平移動 ~0.18 m/s（滾動，不是打滑）；高度只看垂直分量。
  3. 物理只有 60 Hz（Kit 預設，每控制步一次物理步；步態訓練是 dt 0.002 × decimation 10）。新增 `PHYSICS_HZ`（PhysxSceneAPI.timeStepsPerSecond）。
  4. 500 Hz 一開始就翻倒：growin 出生 prim 高 0.55 在零位姿（腿伸直）時腳插進地板 ~10 cm，被彈到 0.87 m。改用通用探測高度。
- 用真值當「理想接觸判斷」的上限：3.1 m 爬升誤差 −0.09 m → 方法可行，瓶頸在接觸判斷。量測力矩 >200 N：precision 97%、recall 89%。
- 最終：垂直 2 狀態 KF（高度、速度），IMU 加速度預測，踩地腳的 −Δh/dt 取中位數當速度量測，3σ 門檻；加速度加 0.5 m/s² 雜訊 + 0.1 偏差仍 <0.15 m。
  60 Hz 錄檔同算法誤差 5~6 m（力矩不準）→ 必須 500 Hz。
- 線上 IMU 加速度先用 articulation 線速度差分，平台上漂 −0.5 m；改用機身高度二次差分（與離線一致）後閉迴路爬 9 m 誤差 −0.34 m，到 3F。

### 10-15 樓梯地圖：在平台一直撞牆
- 第 2 個平台上 approach 第 4 段 600+ 步，前方淨空 0.22 m。原因：該段中線（across）0.9→0.3→1.0 跳動，偏到 x=0.3 時對正點落在平台角落貼牆。
  從平台斜看常只看到樓梯寬度的一部分，中心偏一側，但每次觀測都以 SMOOTH=0.3 拉中線。
- 修正 `_integrate`：權重 = SMOOTH × min(1, 觀測半寬/地圖半寬)²，窄（不完整）的觀測幾乎不動中線。
- 結果（M2F + 500 Hz + HEIGHT=odom）：第 1 趟 4244 步到 3F，平台上沒有再長時間撞牆。
- 同時新增 `/go2/semantic_truth`（M2F=1 時發完美語意給 rviz 對照；ViPlanner 仍吃 M2F），rviz 設定加 SemanticTruth 顯示；M2F=0 時額外相機改用完美語意。

### 10-15 下樓走回頭路 → 上下樓都不走回頭（stair_map + 腿部里程計）
- 走回頭原因：腿部里程計下樓一段就 +1.07 m（剛落地仍在下沉的腳被當靜止；共用 3σ 門檻把正確量測也擋掉），
  加上地圖點用真實相機高度、腳高用里程計 → 一個 4 次觀測的碎片被當成下一段，對正點在剛下來的樓梯上 → 爬回去。
- 修正：
  1. stair_map.update()：IMU 俯仰角往反方向 >15° 連續 8 次 → 丟掉目標（標 walked）、往機身後方走（state "back"）。不依賴高度。
  2. leg_odometry：逐腳判斷「用 IMU 預測的機身速度 + 腳相對速度 ≈ 0（<0.2 m/s）」才算靜止；有受力腳但連續 15 次都不靜止 → 直接採用（避免發散）。
     5 份 500 Hz 錄檔最壞誤差 ~0.58 m（上樓系統性少 ~6%，可能是球形腳在階緣滾動）；參數幾乎不影響。
  3. HEIGHT=odom 時地圖點高度加上里程計誤差（模擬實機：點和腳高同一座標）。
  4. 下樓候選只要求往下 ≥0.6 m（MIN_DROP；從上往下斜看只看得到前幾階，原 1.5 m 讓 2F 找不到下一段、走出場景）。
  5. 同層門檻 = 0.8 + 0.10 × 已走垂直距離（爬 9 m 後里程計仍少 0.88 m，曾在 3F 繼續找樓梯走出場景邊緣）。
- 結果（M2F + 500 Hz + HEIGHT=odom）：下樓 4887 / 5049 步到 1F；上樓 4101 步到 3F；都沒有走回頭。

## 2026-10-21 — growin 在 L/U 型場景啟動卡死：非同步渲染在 PLAY 事件裡切換
症狀：`timeline.play()` 後第一次 app update 永不返回，kit 主執行緒 100% CPU（使用者空間），其餘執行緒全在 futex/sleep。
growin + 用 UsdGeom.Cube 搭的場景（L 型連 3/3、U 型約一半）會中；go2 同場景、growin 在三樓 Stair.usdc 場景都沒中。
查法：主機 ptrace_scope=1、無 sudo、perf_event_paranoid=4，只能在容器裡看。`run_stair_usd.sh` 加 `DEBUG_PTRACE=1`（`--cap-add=SYS_PTRACE`），
py-spy（pip download 的 wheel 拆出 binary，`docker cp` 進去）要用 `docker exec -u 0` 跑（容器以 uid 1000 執行時 capability 不生效），
`py-spy dump --native --pid 1`。主執行緒堆疊：timeline 事件 → `isaacsim.core.throttling/extension.py:117 on_stop_play`
（`_settings.set("/app/asyncRendering", False)`）→ carb.dictionary → libomni.usd 內反覆 lock/unlock。
`isaacsim.exp.full.kit` 設 `exts."isaacsim.core.throttling".enable_async = true`：啟動時開非同步渲染，PLAY 時關、STOP 後再開。
修法（`viplanner_control.run_walk`，play 之前）：`enable_async=False`、`/app/asyncRendering=False`、`asyncRenderingLowLatency=False`，再跑 2 次 app update。
PLAY 事件因此不再切換。為什麼只有 growin + 方塊場景會中，未查明（推測與渲染時序有關）。
