# 現況快照

## 主要目標
把 Go2 機器狗用 ViPlanner（視覺導航）+ HIMLoco（步態 policy）跑起來，自主爬樓梯走到 2F 房間目標點。
現行主線是 Isaac Sim 5.1.0 獨立 Docker 環境 + 自建雙截樓梯場景，Isaac Gym 版（09-09）已不再更新。

## 現況

### 2026-09-16 ~ 10-06 摘要（原文封存在 history.md「2026-10-14 封存」段）
- **09-16/17 自建雙截樓梯場景**：`isaac_sim_bridge/viplanner_control.py`（控制迴圈）+ `isaac-sim-docker/home/open_scene_stairs.py`。
  修好樓梯背面誤標、入口不敢轉彎、landing 卡住等 bug 後 20/20 通過。v3 模型（`..._stairs_finetune_v3`，val 10.14）是之後所有微調的起點。
  即時參數：`REPLAN_EVERY_N_STEPS=8`、`COMMAND_HOLD_STEPS=8`、`HEADING_SMOOTH_RATE=0.4`。`DISABLE_STAIR_SIDE_WALLS` 開關、虛擬邊界牆、高度可走性檢查。
- **09-22 L 型樓梯場景**（`l_shaped_stair_scene_with_walls_viplanner.py`）：單一目標點模式穩定跑通。
  語意原則（用戶確認）：太高不可一步跨上 → 牆；有高度差但可爬 → 樓梯；其餘地板。
- **10-01 Stair.usdc**（`~/Downloads/Stair.usdc`，折返樓梯間 1F→3F，階高 0.18 m）：入口 `open_scene_stair_usd.py`、啟動 `run_stair_usd.sh`
  （`COLLECT=1 EPISODES=n COLLECT_DIR=x` 收集）。Flight3/Landing2 的碰撞經用戶同意寫進原檔（備份 `Stair.usdc.bak`）。
  新語意 RAILING（=fence 顏色，cost 同牆）。跌出場景 / 傾斜 >70° 持續 200 步會結束 episode。v5、v6、v6_cont 已訓練（舊相機、含假牆資料）。
  串接腳本 `training_data/collect_then_train_stair_usd.sh`：收集完自動關 container → 過濾 → 訓練。
- **10-02 往實機推進（四階段）**：①深度語意 `geometric_semantic_v2.py`（`GEO_V2=1`，整體 88%）完成；②無標註 cost map 工具
  （`tools/export_collision_mesh.py`、`build_cloud_from_mesh.py`）完成；③控制不依賴 aabbs（未做）；④實機（未做）。
  相機修正 `CAMERA_NEAR_CLIP=0.2`。控制迴圈跟 Kit 畫面更新跑（約 60 Hz），不是訓練時的 50 Hz。
- **10-06 combine_without_BlendersWall 場景**：`run_combine.sh` → `open_scene_combine.py`。splat 需轉 NuRec 才顯示；
  深度相機看不到 splat → `VIPLANNER_PROXY_DEPTH=1` 用碰撞方塊算深度；桌子當實心。3/3 到達。

### 2026-10-13 — 幽靈狗測試找到搖晃主因、目標點只留 3F、換成 leo
- **幽靈狗模式**（`GHOST=1`，`run_walk` 內）：ViPlanner + 導引照常算出 vx/wz，但機身用運動學直接移動、不經步態；
  每 episode 的路線與每次規劃存到 `training_data/ghost_logs/<name>/`，`tools/run_ghost_test.sh <name> [N]` 一鍵跑完並畫圖
  （`tools/plot_ghost_paths.py`）。`GHOST_HEIGHT` 可調飄浮高度。
- **搖晃主因 = 卡住偵測誤判**（不是步態、不是 ViPlanner）：原本只看「200 步內有沒有更靠近 planning_goal」，爬樓梯時目標是
  「下一階」會跟著往前移 → 每 200 步誤判一次 → 強制最大轉速反向轉 80 步。改成還要「實際位移（含高度）< 0.15 m」。
  幽靈狗前後：總轉向 3310° → 656°、步數 6101 → 3642、卡住觸發 54 → 0。
- **在 Stair.usdc，`seeking_stairs` 100% 為 True → 轉向完全由導引的 planning_goal 決定，ViPlanner 的路徑只被畫出來、
  只有 fear 影響速度**。所以換模型（v3/v5/v6/v7）在此場景幾乎不改變走法；要評估 ViPlanner 本身需另加「轉向聽 ViPlanner」開關（未做）。
- **目標點只留 3F**（用戶要求；`VIA_WAYPOINTS=1` 可加回中繼點）。配合修正：爬完一段樓梯改朝「頂階再往前 0.8 m」走
  （原本朝最終目標 XY，目標在身後時會在樓梯頂掉頭繞圈）。
- **換成 leo**（用戶提供 `viplanner2/1013/`：`leo/urdf/leo_description.urdf`、`policy_3800.pt`、`env_config.json`）：
  `ROBOT=leo`（`run_stair_usd.sh` 預設）/ `go2`。USD 由 `tools/urdf_to_usd.py` 轉到 `isaac-sim-docker/home/robots/leo/`。
  規格：關節順序同 Go2；站姿 (0,±0.6,∓1.3)/(0,±0.7,∓1.3)；KP 970、KD 26.2162；動作比例髖 0.125 其餘 0.25；站高 0.58 m。
  導引邏輯的高度判斷一律換算回 0.30 m 基準（`ROBOT_LOGIC_Z_OFFSET`）。相機（用戶提供實機規格）：位置 (0.51,0,-0.08)、水平朝前不下傾、**直式** 寬 600×高 960、FOV 水平 78°×垂直 105°、
  範圍 0.2~20 m；跟拍鏡頭 2.0/1.05 m。實跑 1 episode：2610 步到 3F、0 次卡住（橫式 960×600/105° 那版也跑過 1 次：2715 步到 3F）。
  注意：模型輸入固定 640×360 橫式，直式影像會被壓扁；既有模型（v3/v7）都是 87°、下傾 8.6°、橫式資料訓練的。
  第一次完整測試（真步態、v3、標準答案語意、只有 3F 目標）：3474 步到 3F、0 次卡住。
- 之後一律用**標準答案語意**（用戶決定，不開 GEO_V2）。`MODEL_NAME=<models 下資料夾>` 可換 ViPlanner 模型。
- v7（`stair_usd_geo_v7`，深度語意、50 ep、從 v3）已訓練完：best val 17.36 @ epoch 28。

- **rviz 即時路線圖 + 深度圖**（10-13 新增、已實測 rviz 端收得到）：
  `/go2/depth_view`（灰階深度，近亮遠暗、無效全黑）、`/go2/route`（Marker：走過路線依高度上色、ViPlanner 路徑依 fear 黃→紅、
  planning_goal、起點、目標、牆/樓梯外框）。rviz 設定另存 `isaac-sim-docker/go2_route.rviz`（俯視），舊的 `go2_pointcloud.rviz` 沒動。
  實測 432 KB 的影像訊息在本機收得到，`make_image_msg` 註解說的 65500B 上限不是硬限制。

### 2026-10-13（下午）— 新增第三隻狗 growin
- 用戶把 leo 的權重與 json 搬到 `1013/leo/`（程式路徑已跟著改），新增 `1013/growin_description/`（URDF、`weight/policy_1001_111828_8940.pt`、`env_config.json`）。
- `ROBOT=growin ./run_stair_usd.sh`。USD 在 `isaac-sim-docker/home/robots/growin/`；腿部碰撞網格（6~10 萬面，物理啟動卡死）換成事先算的凸包 `*_hull.STL`，
  機身碰撞 URDF 本來就是方塊。原始 `1013/` 檔案沒動。
- 設定：KP 1100、KD 25、動作比例髖 0.125 其餘 0.25、站姿 thigh ∓0.6（前）/∓0.7（後）calf -1.3、出生高 0.55、站高 0.53。
  相機取自 `xacro/cameras_real.xacro` 的 cam_front：(0.495,0,0.105)、水平、640×360、87°；xacro 的量測範圍 0.52~6.0 m **尚未套用**（仍用 0.2 m 近端），待用戶決定。

### 2026-10-14 — 實機路線定案：Mask2Former 語意 + 腿部里程計估高度，狗用 growin
- 用戶決定：語意改用 **Mask2Former**（不用深度語意）；高度用**腿部里程計累計**（出發 0，跟目標高度直接比，差距大就找樓梯）；狗用 **growin**。
- 環境：conda `mask2former_env`（py3.10、torch 2.7 cu128、HF transformers；不裝 mmdet）。模擬新增 `SAVE_RGB=1`（配 `COLLECT=1`）存 `rgb/N.png`。
- 測試腳本 `training_data/tools/m2f_test.py`（類別對照用 `viplanner_core/viplanner/config/coco_sem_meta.py`）。資料 `training_data/growin_rgb_test/`（617 張，取 155 張）：
  | COCO panoptic 現成權重 | 樓梯認成樓梯 | 地板認成可走 | 牆認成障礙 | 有樓梯的 63 張中找到 | 速度 (5090, 640×360) |
  |---|---|---|---|---|---|
  | swin-tiny | 68% | 49% | 89% | 39 | 21 ms |
  | swin-large | 87% | 59% | 92% | 56 | 34 ms |
  弱點：地板常被認成牆或沒類別、欄杆 20~26% 被認成樓梯、貼很近的台階有破洞。對照圖在 `growin_rgb_test/m2f_*/`。
- **已接進控制迴圈**：主機先開 `conda run -n mask2former_env python training_data/tools/m2f_server.py`（127.0.0.1:8765，swin-large，log 在 `training_data/logs/m2f_server.log`），
  再 `ROBOT=growin M2F=1 ./run_stair_usd.sh`。只換掉 ViPlanner 看到的語意（及 `/go2/semantic_view`、收集存檔）；導引/防撞仍查 aabbs。來回 34 ms，失敗時該幀退回標準答案語意。
  換地板顏色實驗（離線改像素）：地板 59% → 65~67%，主因是視角（爬樓梯時地板只剩畫面底部窄條），不是顏色。
- **`GUIDANCE=perception`（10-14）**：樓梯子目標改由 `isaac_sim_bridge/stair_perception.py` 算（Mask2Former 樓梯像素 + 深度 → 斜面擬合得方向/坡度 → 入口、對正點、沿中線前進；
  看不到時沿記憶走、機身水平且走出 0.9 m 或前方 <1.1 m 才算走完；找不到時原地左轉）。高度差仍用模擬真值（腿部里程計未做）；防撞/邊緣/卡住偵測仍查 aabbs。
  - **上樓**（`ROBOT=growin M2F=1 GUIDANCE=perception`）：18 趟 16 趟到 3F（4486~6696 步），2 趟卡到上限。
  - **下樓**（`REVERSE=1`：3F 出發、1F 目標，`open_scene_stair_usd.py` 新增）：仍不穩，最好一次走完 3 段；卡在平台（相鄰同向樓梯混淆、走回頭路、找不到下一段）。
  - growin 另外 4 台相機（xacro：front_bottom/left_bottom/right_bottom 下傾 45°、rear 120°）已加，只用來找樓梯（`EXTRA_CAMS=0` 關），rviz `/go2/semantic_extra` 2×2 拼圖；幀率約 13 fps。
  - 下樓限速 0.3 m/s（全速時一次掉 3~4 階卡死）。除錯：`PERCEP_DUMP=<dir>` 存幀到 `training_data/<dir>/`。
- **樓梯地圖 `isaac_sim_bridge/stair_map.py`（10-14，取代 stair_perception.py）**：每幀樓梯點分組擬合斜面，累積成持久地圖（方向/中線/寬/高度/走過與否），挑下一段走。
  rviz 方塊已拿掉（用戶要求）。修正：混合兩段的觀測丟掉、合併前比對平面高度、走反方向高度降 0.3 m 就丟掉該段、同一段被拆兩半會合併、入口改用「平面與腳同高處」、入口卡住重新對正。
  **上樓 3/3 到 3F**（6536/4865/5272 步）。下樓（`REVERSE=1`）用地圖後曾走完 4 段到 1F，但一趟在 2F 跌倒（未查）。
- **腿部里程計（10-14~15）**：`isaac_sim_bridge/leg_odometry.py`：URDF 腿部運動學（已用模擬真實腳位置驗證，誤差 0）+ 關節力矩算腳受力（>200 N 算踩地）
  + **IMU 垂直加速度與腳速度的卡爾曼濾波**（中位數、3σ 門檻剔除）。`HEIGHT=odom` 導航改用估算高度。
  **必須 `PHYSICS_HZ=500`**（步態訓練是 500 Hz；預設 60 Hz 時力矩太粗，誤差 5~6 m）。
  **閉迴路結果（10-15）：只靠腿部里程計高度，上樓 4101 步到 3F、下樓 4887/5049 步到 1F，不再走回頭**（細節 history「上下樓都不走回頭」）。
  里程計上樓系統性少 ~6%（9 m 少 ~0.6 m），同層門檻隨垂直距離放寬（0.8 + 10%）。
  模擬裡的 IMU 加速度 = 機身高度二次差分（articulation 速度讀值會在平台漂 0.5 m，不用）。錄檔 `LEGODOM_DUMP=<名>`（training_data/legodom/，up2_* 為 60 Hz 舊資料可刪）。
- **growin 出生高度修正（10-15）**：`open_scene_stair_usd.py` 原本 growin 寫死 0.55 → 腳插進地板、被彈到 0.87 m 前傾（60 Hz 時前傾 15°，500 Hz 直接翻倒）。改用通用探測（腳離地 2 cm）。
- 尚未做：500 Hz 下導航成功率統計、下樓測試、樓梯地圖的 3D 點仍用真實相機位姿（實機要換成里程計位姿）、實機影像測試。

### 2026-10-16 — 步態改 50 Hz、U 型場景、只用前方相機下樓失敗
- **對照組（60 Hz 步態、500 Hz 物理、odom、4 台額外相機、下樓）100 趟：59 到 1F、27 超時、11 翻倒、3 掉出場景**（log 存 `training_data/logs/launch_growin_down_60hz_2026-10-16.log`）。
- **步態 50 Hz**：原本每次 Kit 更新跑一步 = 60 Hz（訓練是 50 Hz）。新增 `VIPLANNER_CONTROL_HZ`（`CONTROL_HZ=`，預設 50，0 = Kit 預設），
  做法同 `SimulationContext.set_simulation_dt(rendering_dt)`（loop runner manual step + time codes）。三樓場景實測每步 0.0200 s。步數類計時（REPLAN 8 步等）變長 60/50。
- **U 型場景**（`SCENE=ushape ./run_stair_usd.sh` → `open_scene_stairs.py`）：可選 ROBOT、出生改遠角 (6,4)、目標 2F 房間中心 (-8.05,0,3.35)、單位設公尺。
  growin 在這場景隨機卡在 timeline.play() 後第一次 app update（CPU 100%）→ 10-21 已找到原因並修好（見下段）。
- `stair_map.MIN_RISE` 1.5 → 1.0（U 型每段只升 1.5 m）。`EXTRA_CAMS=` 傳進容器。
- **三樓、60 Hz 物理、50 Hz 步態、只前方相機（EXTRA_CAMS=0）、真值高度、下樓：3 趟 0 到達**（摔下樓梯井 / 平台卡住 / 2F 找不到往 1F）。
  只用前方相機從上往下看樓梯點太少；另發現俯仰角守衛會把「在樓梯上轉身」誤判成走錯樓梯。log `launch_growin_down_60phys.log`。
- **同設定改上樓（REVERSE=0）：48 趟 44 到 3F**（中位 ~4280 步）。失敗：18（第 2 段頂端卡住）、38/45（第 1 段入口卡住，
  z 0.40~0.42 整趟不動，重新對正 20+ 次無效）、44（入口滑下 0.31 m 觸發 WRONG_WAY_DROP 把正確的第 1 段丟掉 → blind 數千步）。
  待用戶決定：脫困動作（後退再對正）、WRONG_WAY_DROP 只在爬升 >0.5 m 後生效或改為重新對正、物理 100 Hz 對照。第 4 趟在第 2 段頂端停滯 ~1800 步後脫困；
  第 18 趟同處卡住 3500 步（全速前進但不動），轉身後機頭朝下 52° 卡死到超時（物理卡住，非導航；疑 60 Hz 物理/50 Hz 步態不整除，未驗證）。log `launch_growin_up_60phys.log`。
- Mask2Former：HF `facebook/mask2former-swin-large-coco-panoptic` 快照 `85b5359…`，transformers 4.57.6 / torch 2.7.0+cu128 / FP32，輸入縮成 384×384。
  5090 一張 33 ms（推論 29.5 ms，與模擬共用 GPU 時量）。ViPlanner 規劃：模擬 6.25 次/秒、真實約 2.1 次/秒（模擬約真實 1/3 速）。

### 2026-10-21 — L 型場景接上啟動腳本；growin 啟動卡死找到原因並修好
- `SCENE=lshape ./run_stair_usd.sh` → `viplanner2/l_shaped_stair_scene_with_walls_viplanner.py`（加 `ROBOT=`、`MAX_STEPS=`）；新增 `STOP_ON_Z=`（→ `VIPLANNER_STOP_ON_Z_MATCH`）、
  `DEBUG_PTRACE=1`（加 `--cap-add=SYS_PTRACE`，py-spy 要 `docker exec -u 0`）。用 GUIDANCE=full（L 型坡度 0.33、每段 0.6 m，stair_map 門檻篩掉）、目標維持起點正上方 2F + STOP_ON_Z=1。
- **卡死原因**：full.kit 預設開非同步渲染，throttling 擴充在 PLAY 事件裡關它（`/app/asyncRendering=False`），主執行緒在 libomni.usd 裡空轉、渲染執行緒全睡。
  growin + 方塊搭的場景（L/U）會中，go2 不會；L 型 growin 連 3 次卡。修法：`run_walk` 在 play 前先關非同步渲染 + `enable_async=False`。修後 L 型 2/2、U 型 1/1 正常啟動。
- 新增 `SCENE=straight`（`straight_stair_scene_viplanner.py`，也加 `ROBOT=`/`MAX_STEPS=`，尚未實跑）。U 型目標改到 2F 離門口最遠角 (-10.4,-4.3,3.35)，約 7.9 m。

### 2026-10-22 — 筆記轉 Word
- `機器狗紀錄viplanner` → `機器狗紀錄viplanner.docx`（44 頁，含新章「關於腳步里程」），舊版留 `.docx.bak`。
  轉檔腳本在 session scratchpad `txt2docx.py`（python-docx；圖用 Noto CJK 畫成 PNG，► ◄ 字型沒有→改畫 ▶ ◀）。

## 活躍問題
| 問題 | 狀態 |
|---|---|
| 入口附近有時會有 ~10 秒的左右微調（繞到正確 y 位置對齊走廊）才真正踏進第一階，不是每次都直衝直入 | 不影響最終成功率，暫不處理，之後如果想更乾淨可以再調 |
| waypoint 2 → 3（跨過 landing、進入 Stair B）這段偶爾會拖很久（20 個 episode 裡約 5 個明顯變慢，~2000-3600 步 vs 平常 300-500 步），但最後都有走完，不是卡死；推測是這個路口視覺較複雜，ViPlanner 自己的 fear 訊號偏保守導致放慢速度，不是像入口/landing 那樣的「導引被誤關掉」問題 | 未深入查，成功率不受影響，暫不處理 |
| Isaac Gym 版（`play_stairs_viplanner.py`）路徑寫死 `"viplanner 2"`（多空格）指向不存在資料夾 | 未修，該版本已非主線，暫不處理 |
| 狗跌倒（base z≈0.1）時 `run_walk` 不會偵測、會一直耗到 max_steps 才重置 | 未處理，收集資料時會浪費時間 |
| PyTorch 必須在 `timeline.play()` 跑起來之後才能碰 CUDA，否則第一個 physics step 無聲卡死 | 已解決，`run_walk()` 的程式碼順序刻意维持，未來改動別打亂 |

## 下一步
- **待用戶選：只加開 front_bottom 一台 / 4 台額外相機全開**，再測下樓（三樓場景、60 Hz 物理、50 Hz 步態）。
- 俯仰角守衛誤判（樓梯上轉身）待修；growin 啟動卡死已修（10-21），再多重開幾次確認。
- **500 Hz 物理是否設為預設**（待用戶決定；影響所有模擬結果）。500 Hz 下 stair_map 在樓梯中段曾反覆「重新對正」左右擺約 1500 步（之後自行脫困），待查。
- **leo 上比較 v3 vs v8（10-14，中斷未完成）**：v8 + `planner_steer` 3/3 到 3F（6736/5616/6346 步，`full` 約 2600 步）；v3 + `off` 1 趟跌倒、沒上樓梯
  （`off` 時真正目標在出生點旁 1.2 m、高 9 m，ViPlanner 走到正下方打轉）。v3 `planner_steer`、v8 `off` 尚未跑。排隊腳本範本在 history 無，需重寫。
- v8 資料：`stair_usd_leo` 50 趟 17883 筆（49 趟到 3F）；v8 10-13 18:19 訓完（30 epoch，best val 14.75，`TRAIN_ROBOT_WIDTH=0.54`，cost map 沿用 `stair_usd_geo/maps`）。
  `stair_usd_leo_partial32/`（27 GB，中斷的舊收集）未用，待用戶決定是否刪。
- **growin 過夜測試（10-13~14，GUIDANCE=full、v3，204 趟）**：180 到 3F（88%，步數中位 3544）、19 趟卡到 26000 步上限、5 趟跌倒。卡住原因未查。
- growin 相機範圍（xacro 0.52~6.0 m）是否套用，待用戶決定。
- 深度語意加 RealSense 類雜訊評估；階段 3（局部高度圖取代 aabbs 導引）
