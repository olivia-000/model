"""
DANN v2 — 四個改善方式整合版

改善一：RF 預訓練初始化 Feature Extractor（取代隨機初始化）
改善二：用 RF feature importance 篩選 top-K 特徵（降低 domain alignment 維度）
改善三：Ensemble Teacher-Student（RF + LightGBM + XGBoost 三模型平均 soft label）
改善四：JSD 自適應 λ 調度（lambda_max = 0.5 × exp(-mean_JSD)）

Usage:
  python dann_v2.py --all --epochs 100
  python dann_v2.py --source NYC --target Chicago --epochs 100 --top_k 12
"""

import os, warnings, argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_score, f1_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from scipy.spatial.distance import jensenshannon

warnings.filterwarnings('ignore')

PROC_DIR  = '../data/processed'
MODEL_DIR = '../outputs/models'
EDA_DIR   = '../outputs/eda'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CATEGORIES = ['violent', 'property', 'other']
GRID_SIZE  = 0.01
ALL_FEAT   = [
    'lat_bin','lon_bin','lat_norm','lon_norm',
    'density_pct','violent_pct','entropy_pct','dom_gap_pct',
    'time_slot','is_weekend','log_count',
    'hour_sin','hour_cos','month_sin','month_cos','weekday_sin','weekday_cos',
    'top1_ratio','dominance_gap','entropy',
    'hist_violent','hist_property','hist_other',
    'lag_violent','lag_property','lag_other',
]
# Features to use for JSD calculation (distribution-sensitive)
JSD_FEAT = ['hist_violent','hist_property','hist_other',
            'lag_violent','lag_property','lag_other',
            'entropy','top1_ratio','dominance_gap','log_count']

print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')


# ══════════════════════════════════════════════════════════════════════════════
# 1. Architecture (same as v1)
# ══════════════════════════════════════════════════════════════════════════════

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambda_ * grad, None

class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)

class FeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, feature_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim), nn.ReLU(), nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

class LabelPredictor(nn.Module):
    def __init__(self, feature_dim=64, n_classes=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )
    def forward(self, x):
        return self.net(x)

class DomainClassifier(nn.Module):
    def __init__(self, feature_dim=64, n_domains=2, dropout=0.2):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, n_domains),
        )
    def forward(self, x, lambda_=1.0):
        self.grl.lambda_ = lambda_
        return self.net(self.grl(x))

class DANN(nn.Module):
    def __init__(self, input_dim, feature_dim=64, n_classes=3, n_domains=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, 128, feature_dim)
        self.label_predictor   = LabelPredictor(feature_dim, n_classes)
        self.domain_classifier = DomainClassifier(feature_dim, n_domains)
    def forward(self, x, lambda_=1.0):
        f = self.feature_extractor(x)
        return self.label_predictor(f), self.domain_classifier(f, lambda_), f


# ══════════════════════════════════════════════════════════════════════════════
# 2. Data Loading (same logic as v1, returns numpy arrays)
# ══════════════════════════════════════════════════════════════════════════════

def load_city_data(df_all, city, grid_size=GRID_SIZE):
    city_ev = df_all[df_all['city'] == city].copy()
    city_ev['datetime'] = pd.to_datetime(city_ev['datetime'], errors='coerce')
    city_ev = city_ev.dropna(subset=['datetime'])

    max_date   = city_ev['datetime'].max()
    min_date   = city_ev['datetime'].min()
    total_days = (max_date - min_date).days

    test_start = max_date - pd.Timedelta(days=int(total_days * 0.20))
    val_start  = max_date - pd.Timedelta(days=int(total_days * 0.40))

    train_ev = city_ev[city_ev['datetime'] <  val_start]
    val_ev   = city_ev[(city_ev['datetime'] >= val_start) & (city_ev['datetime'] < test_start)]
    test_ev  = city_ev[city_ev['datetime'] >= test_start]

    def build_grid(events):
        if len(events) == 0:
            return pd.DataFrame()
        d = events.copy()
        d['lat_bin']   = (d['latitude']  / grid_size).round() * grid_size
        d['lon_bin']   = (d['longitude'] / grid_size).round() * grid_size
        d['hour']      = pd.to_datetime(d['datetime']).dt.hour
        d['month']     = pd.to_datetime(d['datetime']).dt.month
        d['weekday']   = pd.to_datetime(d['datetime']).dt.weekday
        d['time_slot'] = pd.cut(d['hour'], bins=[-1,5,11,17,23], labels=[0,1,2,3]).astype(int)
        d['crime_category'] = d['crime_category'].replace({'drug':'other','public_order':'other'})
        for cat in CATEGORIES:
            d[f'cnt_{cat}'] = (d['crime_category'] == cat).astype(int)
        grid = d.groupby(['lat_bin','lon_bin','time_slot']).agg(
            total_count=('crime_category','size'),
            hour_mean=('hour','mean'),
            month_mean=('month','mean'),
            weekday_mean=('weekday','mean'),
            cnt_violent=('cnt_violent','sum'),
            cnt_property=('cnt_property','sum'),
            cnt_other=('cnt_other','sum'),
        ).reset_index()
        cnt_cols = ['cnt_violent','cnt_property','cnt_other']
        for cat in CATEGORIES:
            grid[f'hist_{cat}'] = grid[f'cnt_{cat}'] / grid['total_count']
        grid['dominant_category'] = grid[cnt_cols].idxmax(axis=1).str.replace('cnt_','',regex=False)
        grid['top1_ratio']    = grid[cnt_cols].values.max(axis=1) / grid['total_count']
        sorted_c              = np.sort(grid[cnt_cols].values, axis=1)[:,::-1]
        grid['dominance_gap'] = (sorted_c[:,0] - sorted_c[:,1]) / grid['total_count']
        hist_arr = np.clip(grid[cnt_cols].values / grid['total_count'].values.reshape(-1,1), 1e-9, 1)
        grid['entropy']   = scipy_entropy(hist_arr.T)
        grid['log_count'] = np.log1p(grid['total_count'])
        return grid[grid['total_count'] >= 3].copy()

    grid_train = build_grid(train_ev)
    grid_val   = build_grid(val_ev)
    grid_test  = build_grid(test_ev)

    def add_lag(grid, ref=None):
        src  = ref if ref is not None else grid
        step = grid_size
        for cat in CATEGORIES:
            lags = []
            for _, row in grid.iterrows():
                lat, lon = row['lat_bin'], row['lon_bin']
                mask = (src['lat_bin'].between(lat-step*1.5, lat+step*1.5) &
                        src['lon_bin'].between(lon-step*1.5, lon+step*1.5) &
                        ~((src['lat_bin']==lat) & (src['lon_bin']==lon)))
                nb = src[mask]
                lags.append(nb[f'hist_{cat}'].mean() if len(nb)>0 else row[f'hist_{cat}'])
            grid[f'lag_{cat}'] = lags
        return grid

    print(f'  [{city}] Computing spatial lag...')
    grid_train = add_lag(grid_train)
    grid_val   = add_lag(grid_val,  ref=grid_train)
    grid_test  = add_lag(grid_test, ref=grid_train)

    def make_features(grid, ref=None):
        d = grid.copy().reset_index(drop=True)
        d['hour_sin']    = np.sin(2*np.pi*d['hour_mean']/24)
        d['hour_cos']    = np.cos(2*np.pi*d['hour_mean']/24)
        d['month_sin']   = np.sin(2*np.pi*d['month_mean']/12)
        d['month_cos']   = np.cos(2*np.pi*d['month_mean']/12)
        d['weekday_sin'] = np.sin(2*np.pi*d['weekday_mean']/7)
        d['weekday_cos'] = np.cos(2*np.pi*d['weekday_mean']/7)
        d['is_weekend']  = (d['weekday_mean'] >= 5).astype(float)
        d['lat_norm']    = (d['lat_bin'] - d['lat_bin'].mean()) / (d['lat_bin'].std() + 1e-8)
        d['lon_norm']    = (d['lon_bin'] - d['lon_bin'].mean()) / (d['lon_bin'].std() + 1e-8)
        d['density_pct'] = d['log_count'].rank(pct=True)
        d['violent_pct'] = d['hist_violent'].rank(pct=True)
        d['entropy_pct'] = d['entropy'].rank(pct=True)
        d['dom_gap_pct'] = d['dominance_gap'].rank(pct=True)
        return d[ALL_FEAT].fillna(0)

    X_train = make_features(grid_train)
    X_val   = make_features(grid_val,  ref=grid_train)
    X_test  = make_features(grid_test, ref=grid_train)

    le = LabelEncoder()
    le.fit(CATEGORIES)

    def safe_filter(grid, X):
        grid = grid.reset_index(drop=True)
        X    = X.reset_index(drop=True)
        mask = grid['dominant_category'].isin(set(le.classes_))
        return X[mask].reset_index(drop=True), le.transform(grid['dominant_category'][mask].values)

    X_train, y_train = safe_filter(grid_train, X_train)
    X_val,   y_val   = safe_filter(grid_val,   X_val)
    X_test,  y_test  = safe_filter(grid_test,  X_test)

    print(f'  [{city}] train={len(X_train)}, val={len(X_val)}, test={len(X_test)}')
    return X_train, y_train, X_val, y_val, X_test, y_test, le


# ══════════════════════════════════════════════════════════════════════════════
# 3. 改善四：JSD 自適應 λ 計算
# ══════════════════════════════════════════════════════════════════════════════

def compute_jsd(X_src, X_tgt, feat_cols=None):
    """Compute mean JSD across distribution-sensitive features."""
    if feat_cols is None:
        feat_cols = JSD_FEAT
    jsds = []
    for col in feat_cols:
        if col not in X_src.columns or col not in X_tgt.columns:
            continue
        bins = np.linspace(
            min(X_src[col].min(), X_tgt[col].min()),
            max(X_src[col].max(), X_tgt[col].max()) + 1e-8,
            30
        )
        p, _ = np.histogram(X_src[col], bins=bins, density=True)
        q, _ = np.histogram(X_tgt[col], bins=bins, density=True)
        p = np.clip(p, 1e-10, None); p /= p.sum()
        q = np.clip(q, 1e-10, None); q /= q.sum()
        jsds.append(float(jensenshannon(p, q)))
    mean_jsd = np.mean(jsds) if jsds else 0.5
    return round(mean_jsd, 4), jsds

def adaptive_lambda_max(mean_jsd, base=0.5):
    """改善四：lambda_max = base × exp(-JSD)"""
    return round(base * np.exp(-mean_jsd), 4)

def dann_schedule(epoch, n_epochs, gamma=5.0, lambda_max=0.5):
    p = epoch / n_epochs
    raw = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
    return lambda_max * raw


# ══════════════════════════════════════════════════════════════════════════════
# 4. 改善一：RF 預訓練初始化 Feature Extractor
# ══════════════════════════════════════════════════════════════════════════════

def pretrain_feature_extractor_with_rf(X_train, y_train, input_dim, feature_dim=64,
                                        hidden_dim=128, n_epochs=30, lr=1e-3):
    """
    Train a MLP classifier on source data, then use its first two layers
    to initialize DANN's FeatureExtractor weights.
    """
    print('  [改善一] Pre-training FeatureExtractor with source MLP...')

    # Train a small MLP with same architecture as FeatureExtractor
    mlp = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(hidden_dim, feature_dim),
        nn.BatchNorm1d(feature_dim), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(feature_dim, len(CATEGORIES)),
    ).to(DEVICE)

    cc = Counter(y_train)
    total = len(y_train)
    n_cls = len(CATEGORIES)
    class_w = torch.FloatTensor([total/(n_cls*cc.get(i,1)) for i in range(n_cls)]).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_w)
    opt = optim.AdamW(mlp.parameters(), lr=lr, weight_decay=1e-4)

    Xt = torch.FloatTensor(X_train.values).to(DEVICE)
    yt = torch.LongTensor(y_train).to(DEVICE)
    dl = DataLoader(TensorDataset(Xt, yt), batch_size=256, shuffle=True)

    for epoch in range(n_epochs):
        mlp.train()
        for xb, yb in dl:
            opt.zero_grad()
            criterion(mlp(xb), yb).backward()
            opt.step()

    # Extract weights from first 6 layers (matching FeatureExtractor.net)
    pretrained_state = {}
    layer_map = {
        'net.0.weight': mlp[0].weight,
        'net.0.bias':   mlp[0].bias,
        'net.1.weight': mlp[1].weight,  # BatchNorm
        'net.1.bias':   mlp[1].bias,
        'net.1.running_mean': mlp[1].running_mean,
        'net.1.running_var':  mlp[1].running_var,
        'net.1.num_batches_tracked': mlp[1].num_batches_tracked,
        'net.4.weight': mlp[4].weight,
        'net.4.bias':   mlp[4].bias,
        'net.5.weight': mlp[5].weight,  # BatchNorm
        'net.5.bias':   mlp[5].bias,
        'net.5.running_mean': mlp[5].running_mean,
        'net.5.running_var':  mlp[5].running_var,
        'net.5.num_batches_tracked': mlp[5].num_batches_tracked,
    }
    print(f'  [改善一] Pre-training done. Weights ready for DANN init.')
    return layer_map


# ══════════════════════════════════════════════════════════════════════════════
# 5. 改善二：RF Feature Importance 特徵篩選
# ══════════════════════════════════════════════════════════════════════════════

def get_top_features(X_train, y_train, top_k=12):
    """Train RF on source, return top-k feature names by importance."""
    print(f'  [改善二] Training RF for feature importance (top_k={top_k})...')
    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    rf.fit(X_train.values, y_train)

    imp = pd.Series(rf.feature_importances_, index=ALL_FEAT).sort_values(ascending=False)
    top_feats = imp.head(top_k).index.tolist()

    print(f'  [改善二] Top {top_k} features:')
    for i, (feat, score) in enumerate(imp.head(top_k).items()):
        print(f'    {i+1:2d}. {feat:<20s} {score:.4f}')

    # Save importance
    imp_df = pd.DataFrame({'feature': imp.index, 'importance': imp.values})
    imp_df.to_csv(f'{MODEL_DIR}/rf_feature_importance_source.csv', index=False)

    return top_feats, rf


# ══════════════════════════════════════════════════════════════════════════════
# 6. 改善三：Ensemble Teacher-Student
# ══════════════════════════════════════════════════════════════════════════════

def ensemble_teacher_student(source_city, target_city,
                              X_src_tr, y_src_tr,
                              X_tgt_tr, y_tgt_tr, X_tgt_val, y_tgt_val,
                              X_tgt_te, y_tgt_te,
                              n_epochs=80, batch_size=256, lr=1e-3,
                              temperature=3.0):
    """
    改善三：Train 3 source models (RF + LightGBM + XGBoost),
    average their soft labels, use as teacher for target MLP.
    """
    import lightgbm as lgb
    import xgboost as xgb
    from sklearn.ensemble import RandomForestClassifier

    print(f'\n  [改善三] Ensemble Teacher-Student: {source_city} → {target_city}')

    # Train 3 source models
    cc = Counter(y_src_tr)
    total = len(y_src_tr)
    n_cls = len(CATEGORIES)
    sw = np.array([total/(n_cls*cc[c]) for c in y_src_tr])

    print('    Training RF...')
    rf = RandomForestClassifier(n_estimators=300, max_depth=12,
                                class_weight='balanced', random_state=42, n_jobs=-1)
    rf.fit(X_src_tr.values, y_src_tr)

    print('    Training LightGBM...')
    lgb_model = lgb.LGBMClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.02,
        class_weight='balanced', random_state=42, n_jobs=-1, verbose=-1
    )
    lgb_model.fit(X_src_tr.values, y_src_tr)

    print('    Training XGBoost...')
    xgb_model = xgb.XGBClassifier(
        n_estimators=300, max_depth=6, learning_rate=0.02,
        eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0
    )
    xgb_model.fit(X_src_tr.values, y_src_tr, sample_weight=sw)

    # Generate soft labels on target train
    p_rf  = rf.predict_proba(X_tgt_tr.values)
    p_lgb = lgb_model.predict_proba(X_tgt_tr.values)
    p_xgb = xgb_model.predict_proba(X_tgt_tr.values)
    p_avg = (p_rf + p_lgb + p_xgb) / 3.0

    # Apply temperature scaling for softer labels
    p_temp = np.exp(np.log(p_avg + 1e-8) / temperature)
    p_temp = p_temp / p_temp.sum(axis=1, keepdims=True)
    soft_labels = torch.FloatTensor(p_temp).to(DEVICE)

    # Train student MLP on target with ensemble soft labels
    input_dim = X_tgt_tr.shape[1]
    student = nn.Sequential(
        nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, n_cls),
    ).to(DEVICE)

    Xt = torch.FloatTensor(X_tgt_tr.values).to(DEVICE)
    Xv = torch.FloatTensor(X_tgt_val.values).to(DEVICE)
    yv = torch.LongTensor(y_tgt_val).to(DEVICE)
    Xte= torch.FloatTensor(X_tgt_te.values).to(DEVICE)

    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=1e-4)
    kl_loss   = nn.KLDivLoss(reduction='batchmean')
    ds = TensorDataset(Xt, soft_labels, torch.LongTensor(y_tgt_tr).to(DEVICE))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_f1, best_state = 0.0, None
    for epoch in range(n_epochs):
        student.train()
        for xb, sl, yb in dl:
            optimizer.zero_grad()
            logits = student(xb)
            log_prob = nn.functional.log_softmax(logits / temperature, dim=1)
            loss = kl_loss(log_prob, sl)
            loss.backward()
            optimizer.step()

        if (epoch+1) % 10 == 0:
            student.eval()
            with torch.no_grad():
                preds_v = student(Xv).argmax(1).cpu().numpy()
            f1_v = f1_score(yv.cpu().numpy(), preds_v, average='macro', zero_division=0)
            if f1_v > best_f1:
                best_f1 = f1_v
                best_state = {k: v.clone() for k, v in student.state_dict().items()}

    if best_state:
        student.load_state_dict(best_state)

    student.eval()
    with torch.no_grad():
        preds = student(Xte).argmax(1).cpu().numpy()

    pm = precision_score(y_tgt_te, preds, average='macro',    zero_division=0)
    pw = precision_score(y_tgt_te, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_tgt_te,        preds, average='macro',    zero_division=0)

    print(f'\n  ===== Ensemble T-S [{source_city}→{target_city}] =====')
    print(f'  Precision macro    = {pm:.4f}')
    print(f'  Precision weighted = {pw:.4f}')
    print(f'  F1 macro           = {f1:.4f}')
    return pm, pw, f1


# ══════════════════════════════════════════════════════════════════════════════
# 7. DANN v2 主訓練函數（整合改善一、二、四）
# ══════════════════════════════════════════════════════════════════════════════

def make_infinite_loader(dataset, batch_size):
    while True:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        for batch in loader:
            yield batch

def to_tensor(X, y=None):
    xt = torch.FloatTensor(X.values if hasattr(X, 'values') else X).to(DEVICE)
    if y is not None:
        return xt, torch.LongTensor(y).to(DEVICE)
    return xt

def train_dann_v2(source_city, target_city,
                  X_src_tr, y_src_tr, X_src_val, y_src_val,
                  X_tgt_tr, y_tgt_tr, X_tgt_val, y_tgt_val,
                  X_tgt_te, y_tgt_te,
                  top_feats=None,
                  n_epochs=100, batch_size=256, lr=1e-3, weight_decay=1e-4):

    n_classes = len(CATEGORIES)

    # 改善二：特徵篩選
    if top_feats is not None:
        X_src_tr  = X_src_tr[top_feats]
        X_src_val = X_src_val[top_feats]
        X_tgt_tr  = X_tgt_tr[top_feats]
        X_tgt_val = X_tgt_val[top_feats]
        X_tgt_te  = X_tgt_te[top_feats]
        print(f'  [改善二] Using {len(top_feats)} features: {top_feats}')

    input_dim = X_src_tr.shape[1]

    # 改善四：計算 JSD → 自適應 lambda_max
    mean_jsd, feat_jsds = compute_jsd(X_src_tr, X_tgt_tr,
                                       feat_cols=[f for f in JSD_FEAT if f in X_src_tr.columns])
    lm = adaptive_lambda_max(mean_jsd)
    print(f'  [改善四] mean_JSD={mean_jsd:.4f} → lambda_max={lm:.4f}')

    # 改善一：預訓練 FeatureExtractor
    pretrained_weights = pretrain_feature_extractor_with_rf(
        X_src_tr, y_src_tr, input_dim=input_dim
    )

    # Build DANN model
    model = DANN(input_dim=input_dim, feature_dim=64,
                 n_classes=n_classes, n_domains=2).to(DEVICE)

    # 改善一：載入預訓練權重
    fe_state = model.feature_extractor.state_dict()
    for k, v in pretrained_weights.items():
        if k in fe_state and fe_state[k].shape == v.shape:
            fe_state[k] = v.clone()
    model.feature_extractor.load_state_dict(fe_state)
    print(f'  [改善一] FeatureExtractor initialized from pre-trained MLP.')

    # Class weights
    cc = Counter(y_src_tr)
    total = len(y_src_tr)
    class_w = torch.FloatTensor([
        total / (n_classes * cc.get(i, 1)) for i in range(n_classes)
    ]).to(DEVICE)

    criterion_class  = nn.CrossEntropyLoss(weight=class_w)
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    Xs_t, ys_t = to_tensor(X_src_tr, y_src_tr)
    Xt_t, yt_t = to_tensor(X_tgt_tr, y_tgt_tr)
    ds_src = TensorDataset(Xs_t, ys_t, torch.zeros(len(Xs_t), dtype=torch.long).to(DEVICE))
    ds_tgt = TensorDataset(Xt_t, yt_t, torch.ones( len(Xt_t), dtype=torch.long).to(DEVICE))

    steps_per_epoch = max(len(X_src_tr), len(X_tgt_tr)) // batch_size
    steps_per_epoch = max(steps_per_epoch, 1)
    inf_src = make_infinite_loader(ds_src, batch_size)
    inf_tgt = make_infinite_loader(ds_tgt, batch_size)

    Xtv_t, ytv_t = to_tensor(X_tgt_val, y_tgt_val)
    Xte_t, yte_t = to_tensor(X_tgt_te,  y_tgt_te)

    best_val_f1, best_state = 0.0, None
    patience, no_improve = 20, 0

    print(f'\n  Training DANN v2: {source_city} → {target_city}')
    print(f'  Source: {len(X_src_tr):,}  Target: {len(X_tgt_tr):,}  Steps/epoch: {steps_per_epoch}')

    for epoch in range(n_epochs):
        model.train()
        lambda_ = dann_schedule(epoch, n_epochs, gamma=5.0, lambda_max=lm)
        total_loss = 0.0

        for _ in range(steps_per_epoch):
            xs, ys_lab, ys_dom = next(inf_src)
            xt, yt_lab, yt_dom = next(inf_tgt)
            optimizer.zero_grad()

            cls_src, dom_src, _ = model(xs, lambda_)
            loss_cls = criterion_class(cls_src, ys_lab)
            loss_dom = criterion_domain(dom_src, ys_dom)
            _, dom_tgt, _ = model(xt, lambda_)
            loss_dom += criterion_domain(dom_tgt, yt_dom)
            loss = loss_cls + 0.3 * loss_dom
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                preds_v = model(Xtv_t)[0].argmax(1).cpu().numpy()
            f1_v = f1_score(ytv_t.cpu().numpy(), preds_v, average='macro', zero_division=0)
            pm_v = precision_score(ytv_t.cpu().numpy(), preds_v, average='macro', zero_division=0)
            avg_loss = total_loss / steps_per_epoch
            print(f'  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  λ={lambda_:.3f}  '
                  f'val_F1={f1_v:.4f}  val_P={pm_v:.4f}')

            if f1_v > best_val_f1:
                best_val_f1 = f1_v
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'  Early stop at epoch {epoch+1}')
                    break

    if best_state:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        preds = model(Xte_t)[0].argmax(1).cpu().numpy()

    pm = precision_score(y_tgt_te, preds, average='macro',    zero_division=0)
    pw = precision_score(y_tgt_te, preds, average='weighted', zero_division=0)
    f1 = f1_score(y_tgt_te,        preds, average='macro',    zero_division=0)

    le = LabelEncoder(); le.fit(CATEGORIES)
    print(f'\n  ===== DANN v2 [{source_city}→{target_city}] TEST =====')
    print(f'  Precision macro    = {pm:.4f}')
    print(f'  Precision weighted = {pw:.4f}')
    print(f'  F1 macro           = {f1:.4f}')
    print(f'  JSD={mean_jsd:.4f}  lambda_max={lm:.4f}')
    print(classification_report(y_tgt_te, preds, target_names=le.classes_, zero_division=0))

    save_path = f'{MODEL_DIR}/dann_v2_{source_city.lower()}_{target_city.lower()}.pt'
    torch.save(model.state_dict(), save_path)
    print(f'  Saved: {save_path}')

    return pm, pw, f1, mean_jsd


# ══════════════════════════════════════════════════════════════════════════════
# 8. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source',  type=str, default=None)
    parser.add_argument('--target',  type=str, default=None)
    parser.add_argument('--all',     action='store_true')
    parser.add_argument('--epochs',  type=int, default=100)
    parser.add_argument('--batch',   type=int, default=256)
    parser.add_argument('--top_k',   type=int, default=12,
                        help='Number of top features to use (改善二). 0 = use all 26.')
    parser.add_argument('--no_ensemble', action='store_true',
                        help='Skip ensemble teacher-student (改善三)')
    args = parser.parse_args()

    print('Loading all_cities.csv...')
    df = pd.read_csv(f'{PROC_DIR}/all_cities.csv', low_memory=False)
    df = df.dropna(subset=['latitude','longitude','crime_category'])
    df = df[df['crime_category'].isin(CATEGORIES)]
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    print(f'Total: {len(df):,}')
    print(df['city'].value_counts().to_string())

    available_cities = df['city'].unique().tolist()

    if args.all:
        pairs = [(s, t) for s in available_cities for t in available_cities if s != t]
    elif args.source and args.target:
        pairs = [(args.source, args.target)]
    else:
        pairs = [('NYC', t) for t in available_cities if t != 'NYC']

    print('\nBuilding grid features for all cities...')
    city_data = {}
    for city in available_cities:
        print(f'\n[{city}]')
        try:
            city_data[city] = load_city_data(df, city)
        except Exception as e:
            print(f'  ERROR: {e}')

    # 改善二：取得 top_k 特徵（用第一個 source 城市的 RF）
    top_feats = None
    if args.top_k > 0 and pairs:
        first_source = pairs[0][0]
        if first_source in city_data:
            X_s, y_s = city_data[first_source][0], city_data[first_source][1]
            top_feats, _ = get_top_features(X_s, y_s, top_k=args.top_k)

    results = []
    for source_city, target_city in pairs:
        if source_city not in city_data or target_city not in city_data:
            print(f'Skipping {source_city}→{target_city}: data not available')
            continue

        X_src_tr, y_src_tr, X_src_val, y_src_val, X_src_te, y_src_te, _ = city_data[source_city]
        X_tgt_tr, y_tgt_tr, X_tgt_val, y_tgt_val, X_tgt_te, y_tgt_te, _ = city_data[target_city]

        print(f'\n{"="*60}')
        print(f'  {source_city} → {target_city}')
        print(f'{"="*60}')

        # DANN v2（改善一 + 二 + 四）
        pm_dann, pw_dann, f1_dann, jsd = train_dann_v2(
            source_city, target_city,
            X_src_tr, y_src_tr, X_src_val, y_src_val,
            X_tgt_tr, y_tgt_tr, X_tgt_val, y_tgt_val,
            X_tgt_te, y_tgt_te,
            top_feats=top_feats,
            n_epochs=args.epochs, batch_size=args.batch
        )

        row = {
            'Source': source_city, 'Target': target_city,
            'JSD': jsd,
            'DANN_v2_P':  round(pm_dann, 4),
            'DANN_v2_F1': round(f1_dann, 4),
        }

        # 改善三：Ensemble Teacher-Student
        if not args.no_ensemble:
            try:
                import lightgbm, xgboost
                pm_ens, pw_ens, f1_ens = ensemble_teacher_student(
                    source_city, target_city,
                    X_src_tr, y_src_tr,
                    X_tgt_tr, y_tgt_tr, X_tgt_val, y_tgt_val,
                    X_tgt_te, y_tgt_te,
                    n_epochs=args.epochs
                )
                row['Ensemble_TS_P']  = round(pm_ens, 4)
                row['Ensemble_TS_F1'] = round(f1_ens, 4)
            except ImportError:
                print('  lightgbm/xgboost not available, skipping ensemble TS')

        results.append(row)

    if results:
        res_df = pd.DataFrame(results)
        print('\n' + '='*70)
        print('DANN v2 RESULTS SUMMARY')
        print('='*70)
        print(res_df.to_string(index=False))
        save_path = f'{MODEL_DIR}/dann_v2_results.csv'
        res_df.to_csv(save_path, index=False)
        print(f'\nSaved: {save_path}')

if __name__ == '__main__':
    main()
