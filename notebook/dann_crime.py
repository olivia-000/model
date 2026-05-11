"""
Domain Adversarial Neural Network (DANN) for Cross-City Crime Transfer Learning
Reference: Ganin et al., 2016 - Domain-Adversarial Training of Neural Networks

Architecture:
  - Feature Extractor F:  26-dim input -> 128 -> 64 hidden features
  - Label Predictor C:    64 -> 32 -> 3 (crime categories)
  - Domain Classifier D:  64 -> 32 -> N_cities (domain adversarial)

Training objective:
  L = L_class(F, C) - lambda * L_domain(F, D)
  (Gradient Reversal Layer flips gradient sign for domain classifier)

Usage:
  python dann_crime.py --source NYC --target Chicago
  python dann_crime.py --source NYC --target London
  python dann_crime.py --all   # run all city pair combinations
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
from collections import Counter

warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
PROC_DIR  = '../data/processed'
MODEL_DIR = '../outputs/models'
EDA_DIR   = '../outputs/eda'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)

DEVICE     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CATEGORIES = ['violent', 'property', 'other']
GRID_SIZE  = 0.01

print(f'Device: {DEVICE}')
if DEVICE.type == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')


# ══════════════════════════════════════════════════════════════════════════════
# 1. DANN Architecture
# ══════════════════════════════════════════════════════════════════════════════

class GradientReversalFunction(torch.autograd.Function):
    """Reverses gradient sign during backprop (the core trick of DANN)."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.lambda_ = 1.0

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.lambda_)


class FeatureExtractor(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=128, feature_dim=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.BatchNorm1d(feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class LabelPredictor(nn.Module):
    def __init__(self, feature_dim=64, n_classes=3, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class DomainClassifier(nn.Module):
    def __init__(self, feature_dim=64, n_domains=2, dropout=0.2):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_domains),
        )

    def forward(self, x, lambda_=1.0):
        self.grl.lambda_ = lambda_
        return self.net(self.grl(x))


class DANN(nn.Module):
    def __init__(self, input_dim=26, feature_dim=64, n_classes=3, n_domains=2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_dim, 128, feature_dim)
        self.label_predictor   = LabelPredictor(feature_dim, n_classes)
        self.domain_classifier = DomainClassifier(feature_dim, n_domains)

    def forward(self, x, lambda_=1.0):
        features      = self.feature_extractor(x)
        class_logits  = self.label_predictor(features)
        domain_logits = self.domain_classifier(features, lambda_)
        return class_logits, domain_logits, features


# ══════════════════════════════════════════════════════════════════════════════
# 2. Data Loading & Preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def load_city_data(df_all, city, grid_size=GRID_SIZE):
    """Build grid features for a city using 60/20/20 temporal split."""
    from scipy.stats import entropy as scipy_entropy

    city_events = df_all[df_all['city'] == city].copy()
    city_events['datetime'] = pd.to_datetime(city_events['datetime'], errors='coerce')
    city_events = city_events.dropna(subset=['datetime'])

    max_date   = city_events['datetime'].max()
    min_date   = city_events['datetime'].min()
    total_days = (max_date - min_date).days

    test_start = max_date - pd.Timedelta(days=int(total_days * 0.20))
    val_start  = max_date - pd.Timedelta(days=int(total_days * 0.40))

    train_ev = city_events[city_events['datetime'] <  val_start]
    val_ev   = city_events[(city_events['datetime'] >= val_start) &
                            (city_events['datetime'] <  test_start)]
    test_ev  = city_events[city_events['datetime'] >= test_start]

    def build_grid(events, ref_grid=None):
        if len(events) == 0:
            return pd.DataFrame()
        d = events.copy()
        d['lat_bin'] = (d['latitude']  / grid_size).round() * grid_size
        d['lon_bin'] = (d['longitude'] / grid_size).round() * grid_size
        d['hour']    = pd.to_datetime(d['datetime']).dt.hour
        d['time_slot'] = pd.cut(d['hour'], bins=[-1,5,11,17,23],
                                labels=[0,1,2,3]).astype(int)
        d['crime_category'] = d['crime_category'].replace(
            {'drug': 'other', 'public_order': 'other'})
        for cat in CATEGORIES:
            d[f'cnt_{cat}'] = (d['crime_category'] == cat).astype(int)
        agg = {
            'total_count':  ('crime_category', 'size'),
            'hour_mean':    ('hour', 'mean'),
            'month_mean':   (pd.to_datetime(d['datetime']).dt.month.rename('month'), 'mean'),
            'weekday_mean': (pd.to_datetime(d['datetime']).dt.weekday.rename('weekday'), 'mean'),
        }
        for cat in CATEGORIES:
            agg[f'cnt_{cat}'] = (f'cnt_{cat}', 'sum')

        d['month']   = pd.to_datetime(d['datetime']).dt.month
        d['weekday'] = pd.to_datetime(d['datetime']).dt.weekday

        grid = d.groupby(['lat_bin','lon_bin','time_slot']).agg(
            total_count=('crime_category', 'size'),
            hour_mean=('hour', 'mean'),
            month_mean=('month', 'mean'),
            weekday_mean=('weekday', 'mean'),
            cnt_violent=('cnt_violent', 'sum'),
            cnt_property=('cnt_property', 'sum'),
            cnt_other=('cnt_other', 'sum'),
        ).reset_index()

        cnt_cols = [f'cnt_{c}' for c in CATEGORIES]
        for cat in CATEGORIES:
            grid[f'hist_{cat}'] = grid[f'cnt_{cat}'] / grid['total_count']
        grid['dominant_category'] = (
            grid[cnt_cols].idxmax(axis=1).str.replace('cnt_', '', regex=False))
        grid['top1_ratio']    = grid[cnt_cols].values.max(axis=1) / grid['total_count']
        sorted_c              = np.sort(grid[cnt_cols].values, axis=1)[:,::-1]
        grid['dominance_gap'] = (sorted_c[:,0] - sorted_c[:,1]) / grid['total_count']
        hist_arr = np.clip(grid[cnt_cols].values / grid['total_count'].values.reshape(-1,1), 1e-9, 1)
        grid['entropy']    = scipy_entropy(hist_arr.T)
        grid['log_count']  = np.log1p(grid['total_count'])
        grid = grid[grid['total_count'] >= 3].copy()
        return grid

    grid_train = build_grid(train_ev)
    grid_val   = build_grid(val_ev,   ref_grid=grid_train)
    grid_test  = build_grid(test_ev,  ref_grid=grid_train)

    # Spatial lag
    def add_lag(grid, ref=None):
        src = ref if ref is not None else grid
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

    FEAT_COLS = [
        'lat_bin','lon_bin',
        'log_count','top1_ratio','dominance_gap','entropy',
        'hist_violent','hist_property','hist_other',
        'lag_violent','lag_property','lag_other',
    ]

    def make_features(grid, ref=None):
        d = grid.copy().reset_index(drop=True)
        src = ref if ref is not None else d

        if ref is not None:
            hist_cols = ['lat_bin','lon_bin','time_slot'] + \
                        [f'hist_{c}' for c in CATEGORIES] + \
                        [f'lag_{c}'  for c in CATEGORIES] + \
                        ['log_count','top1_ratio','dominance_gap','entropy']
            avail = [c for c in hist_cols if c in src.columns]
            drop  = [c for c in avail if c not in ['lat_bin','lon_bin','time_slot']]
            d = d.drop(columns=[c for c in drop if c in d.columns])
            d = d.merge(src[avail], on=['lat_bin','lon_bin','time_slot'], how='left')
            for c in CATEGORIES:
                med = src[f'hist_{c}'].median()
                d[f'hist_{c}'] = d[f'hist_{c}'].fillna(med)
                d[f'lag_{c}']  = d.get(f'lag_{c}', pd.Series([med]*len(d))).fillna(med)

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

        ALL_FEAT = [
            'lat_bin','lon_bin','lat_norm','lon_norm',
            'density_pct','violent_pct','entropy_pct','dom_gap_pct',
            'time_slot','is_weekend','log_count',
            'hour_sin','hour_cos','month_sin','month_cos','weekday_sin','weekday_cos',
            'top1_ratio','dominance_gap','entropy',
            'hist_violent','hist_property','hist_other',
            'lag_violent','lag_property','lag_other',
        ]
        return d[ALL_FEAT].fillna(0)

    X_train = make_features(grid_train)
    X_val   = make_features(grid_val,   ref=grid_train)
    X_test  = make_features(grid_test,  ref=grid_train)

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
# 3. Training
# ══════════════════════════════════════════════════════════════════════════════

def to_tensor(X, y=None):
    xt = torch.FloatTensor(X.values).to(DEVICE)
    if y is not None:
        yt = torch.LongTensor(y).to(DEVICE)
        return xt, yt
    return xt


def dann_schedule(epoch, n_epochs, gamma=5.0, lambda_max=0.5):
    """GRL lambda schedule: 0 -> lambda_max over training.
    
    Fix 1: Lower gamma (5.0 vs 10.0) = slower ramp-up
    Fix 2: lambda_max=0.5 prevents domain classifier from dominating
    """
    p = epoch / n_epochs
    raw = 2.0 / (1.0 + np.exp(-gamma * p)) - 1.0
    return lambda_max * raw


def make_infinite_loader(dataset, batch_size):
    """Infinite DataLoader — avoids truncation when sizes are unequal."""
    while True:
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=True, drop_last=True)
        for batch in loader:
            yield batch


def train_dann(source_city, target_city,
               X_src_tr, y_src_tr, X_src_val, y_src_val,
               X_tgt_tr, y_tgt_tr, X_tgt_val, y_tgt_val,
               X_tgt_te, y_tgt_te,
               n_epochs=80, batch_size=256, lr=1e-3, weight_decay=1e-4):

    n_classes = len(CATEGORIES)
    n_domains = 2  # source=0, target=1

    model = DANN(input_dim=26, feature_dim=64,
                 n_classes=n_classes, n_domains=n_domains).to(DEVICE)

    # Class weights for label predictor
    cc = Counter(y_src_tr)
    total = len(y_src_tr)
    class_w = torch.FloatTensor([
        total / (n_classes * cc.get(i, 1)) for i in range(n_classes)
    ]).to(DEVICE)

    criterion_class  = nn.CrossEntropyLoss(weight=class_w)
    criterion_domain = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Fix 3: Infinite loaders — both cities see equal batches per epoch
    # Steps per epoch = max(src_size, tgt_size) / batch_size
    Xs_t, ys_t = to_tensor(X_src_tr, y_src_tr)
    Xt_t, yt_t = to_tensor(X_tgt_tr, y_tgt_tr)

    ds_src = TensorDataset(Xs_t, ys_t,
                           torch.zeros(len(Xs_t), dtype=torch.long).to(DEVICE))
    ds_tgt = TensorDataset(Xt_t, yt_t,
                           torch.ones(len(Xt_t), dtype=torch.long).to(DEVICE))

    steps_per_epoch = max(len(X_src_tr), len(X_tgt_tr)) // batch_size
    steps_per_epoch = max(steps_per_epoch, 1)

    inf_src = make_infinite_loader(ds_src, batch_size)
    inf_tgt = make_infinite_loader(ds_tgt, batch_size)

    Xsv_t, ysv_t = to_tensor(X_src_val, y_src_val)
    Xtv_t, ytv_t = to_tensor(X_tgt_val, y_tgt_val)
    Xte_t, yte_t = to_tensor(X_tgt_te,  y_tgt_te)

    best_val_f1, best_state = 0.0, None
    # Fix 4: Early stopping patience = 20 checkpoints (every 5 epochs)
    patience, no_improve = 20, 0

    print(f'\n  Training DANN: {source_city} → {target_city}')
    print(f'  Source train: {len(X_src_tr):,}  Target train: {len(X_tgt_tr):,}')
    print(f'  Steps/epoch: {steps_per_epoch}')

    for epoch in range(n_epochs):
        model.train()
        lambda_ = dann_schedule(epoch, n_epochs)

        total_loss = 0.0
        for _ in range(steps_per_epoch):
            xs, ys_lab, ys_dom = next(inf_src)
            xt, yt_lab, yt_dom = next(inf_tgt)

            optimizer.zero_grad()

            # Source: class loss + domain loss
            cls_src, dom_src, _ = model(xs, lambda_)
            loss_cls = criterion_class(cls_src, ys_lab)
            loss_dom = criterion_domain(dom_src, ys_dom)

            # Target: domain loss only (no label supervision)
            _, dom_tgt, _ = model(xt, lambda_)
            loss_dom += criterion_domain(dom_tgt, yt_dom)

            # Fix 5: Domain weight 0.3 (was 0.5) — class loss dominates more
            loss = loss_cls + 0.3 * loss_dom
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        # Validation every 5 epochs
        if (epoch + 1) % 5 == 0:
            model.eval()
            with torch.no_grad():
                logits_tgt_val, _, _ = model(Xtv_t)
                preds_val = logits_tgt_val.argmax(dim=1).cpu().numpy()
                f1_val = f1_score(ytv_t.cpu().numpy(), preds_val,
                                  average='macro', zero_division=0)
                pm_val = precision_score(ytv_t.cpu().numpy(), preds_val,
                                         average='macro', zero_division=0)

            avg_loss = total_loss / steps_per_epoch
            print(f'  Epoch {epoch+1:3d}/{n_epochs}  loss={avg_loss:.4f}  '
                  f'λ={lambda_:.3f}  val_F1={f1_val:.4f}  val_P={pm_val:.4f}')

            if f1_val > best_val_f1:
                best_val_f1 = f1_val
                best_state  = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve  = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f'  Early stop at epoch {epoch+1}')
                    break

    # Load best model
    if best_state:
        model.load_state_dict(best_state)

    # Test evaluation
    model.eval()
    with torch.no_grad():
        logits_te, _, _ = model(Xte_t)
        preds_te = logits_te.argmax(dim=1).cpu().numpy()

    pm = precision_score(y_tgt_te, preds_te, average='macro',    zero_division=0)
    pw = precision_score(y_tgt_te, preds_te, average='weighted', zero_division=0)
    f1 = f1_score(y_tgt_te,        preds_te, average='macro',    zero_division=0)

    print(f'\n  ===== DANN [{source_city}→{target_city}] TEST =====')
    print(f'  Precision macro    = {pm:.4f}')
    print(f'  Precision weighted = {pw:.4f}')
    print(f'  F1 macro           = {f1:.4f}')

    le = LabelEncoder()
    le.fit(CATEGORIES)
    print(classification_report(y_tgt_te, preds_te,
                                 target_names=le.classes_, zero_division=0))

    # Save model
    save_path = f'{MODEL_DIR}/dann_{source_city.lower()}_{target_city.lower()}.pt'
    torch.save(model.state_dict(), save_path)
    print(f'  Saved: {save_path}')

    return pm, pw, f1


# ══════════════════════════════════════════════════════════════════════════════
# 4. Baseline (source-only, no adversarial)
# ══════════════════════════════════════════════════════════════════════════════

def train_source_only(source_city, target_city,
                      X_src_tr, y_src_tr,
                      X_tgt_te, y_tgt_te,
                      n_epochs=80, batch_size=256, lr=1e-3):
    """Train on source only, test on target (no domain adversarial)."""
    model = nn.Sequential(
        nn.Linear(26, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(64, 32),  nn.ReLU(),
        nn.Linear(32, len(CATEGORIES)),
    ).to(DEVICE)

    cc = Counter(y_src_tr)
    total = len(y_src_tr)
    class_w = torch.FloatTensor([
        total / (len(CATEGORIES) * cc.get(i, 1)) for i in range(len(CATEGORIES))
    ]).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_w)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    Xs_t, ys_t = to_tensor(X_src_tr, y_src_tr)
    dl = DataLoader(TensorDataset(Xs_t, ys_t), batch_size=batch_size, shuffle=True)
    Xte_t, yte_t = to_tensor(X_tgt_te, y_tgt_te)

    for epoch in range(n_epochs):
        model.train()
        for xb, yb in dl:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = model(Xte_t).argmax(dim=1).cpu().numpy()

    pm = precision_score(y_tgt_te, preds, average='macro',    zero_division=0)
    f1 = f1_score(y_tgt_te,        preds, average='macro',    zero_division=0)
    print(f'  [Source-only NN] {source_city}→{target_city}: P={pm:.4f}  F1={f1:.4f}')
    return pm, f1


# ══════════════════════════════════════════════════════════════════════════════
# 5. Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None)
    parser.add_argument('--target', type=str, default=None)
    parser.add_argument('--all',    action='store_true')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch',  type=int, default=256)
    args = parser.parse_args()

    # Load data
    print('Loading all_cities.csv...')
    df = pd.read_csv(f'{PROC_DIR}/all_cities.csv', low_memory=False)
    df = df.dropna(subset=['latitude','longitude','crime_category'])
    df = df[df['crime_category'].isin(CATEGORIES)]
    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
    df = df.dropna(subset=['datetime'])
    print(f'Total: {len(df):,}')
    print(df['city'].value_counts().to_string())

    available_cities = df['city'].unique().tolist()

    # City pairs to run
    if args.all:
        pairs = [(s, t) for s in available_cities
                         for t in available_cities if s != t]
    elif args.source and args.target:
        pairs = [(args.source, args.target)]
    else:
        # Default: NYC as source to all others
        pairs = [('NYC', t) for t in available_cities if t != 'NYC']

    # Pre-load all city data
    print('\nBuilding grid features for all cities...')
    city_data = {}
    for city in available_cities:
        print(f'\n[{city}]')
        try:
            city_data[city] = load_city_data(df, city)
        except Exception as e:
            print(f'  ERROR: {e}')

    # Run DANN for each pair
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

        # Baseline: source-only NN (no domain adaptation)
        pm_so, f1_so = train_source_only(
            source_city, target_city,
            X_src_tr, y_src_tr,
            X_tgt_te, y_tgt_te,
            n_epochs=args.epochs, batch_size=args.batch
        )

        # DANN
        pm_dann, pw_dann, f1_dann = train_dann(
            source_city, target_city,
            X_src_tr, y_src_tr, X_src_val, y_src_val,
            X_tgt_tr, y_tgt_tr, X_tgt_val, y_tgt_val,
            X_tgt_te, y_tgt_te,
            n_epochs=args.epochs, batch_size=args.batch
        )

        results.append({
            'Source':          source_city,
            'Target':          target_city,
            'Source-only P':   round(pm_so,   4),
            'Source-only F1':  round(f1_so,   4),
            'DANN P':          round(pm_dann,  4),
            'DANN F1':         round(f1_dann,  4),
            'Improvement P':   round(pm_dann - pm_so, 4),
            'Improvement F1':  round(f1_dann - f1_so, 4),
        })

    # Summary
    if results:
        res_df = pd.DataFrame(results)
        print('\n' + '='*70)
        print('DANN RESULTS SUMMARY')
        print('='*70)
        print(res_df.to_string(index=False))
        save_path = f'{MODEL_DIR}/dann_results.csv'
        res_df.to_csv(save_path, index=False)
        print(f'\nSaved: {save_path}')


if __name__ == '__main__':
    main()
