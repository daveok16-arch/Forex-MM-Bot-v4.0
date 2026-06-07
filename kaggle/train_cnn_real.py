import os, sys, numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class CNNPattern(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1)
        )
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 3))
    def forward(self, x):
        x = self.conv(x.unsqueeze(1))
        x = x.view(x.size(0), -1)
        return torch.softmax(self.fc(x), dim=-1)
def label_window(prices):
    if len(prices) < 60: return 2
    first = prices[0]
    last = prices[-1]
    high = max(prices)
    low = min(prices)
    range_pct = (high - low) / first * 100
    change_pct = (last - first) / first * 100
    if change_pct > 0.3 and range_pct > 0.5: return 0
    if change_pct < -0.3 and range_pct > 0.5: return 1
    return 2
class RealDataset(Dataset):
    def __init__(self, csv_path, window=60):
        df = pd.read_csv(csv_path)
        self.samples = []
        prices = df["Close"].values
        for i in range(0, len(prices) - window, 5):
            w = prices[i:i+window]
            if len(w) == window:
                p = (w - w.mean()) / (w.std() + 1e-8)
                label = label_window(w)
                self.samples.append((p.astype(np.float32), label))
        print(f"[Dataset] {len(self.samples)} samples from {len(prices)} bars")
    def __len__(self): return len(self.samples)
    def __getitem__(self, i):
        p, y = self.samples[i]
        return torch.from_numpy(p), torch.tensor(y, dtype=torch.long)
def train(csv_path, out_dir):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[CNN] {dev}")
    ds = RealDataset(csv_path)
    train_size = int(0.8 * len(ds))
    val_size = len(ds) - train_size
    train_ds, val_ds = torch.utils.data.random_split(ds, [train_size, val_size])
    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=2)
    val_dl = DataLoader(val_ds, batch_size=64)
    m = CNNPattern().to(dev)
    opt = optim.Adam(m.parameters(), lr=0.001, weight_decay=1e-5)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)
    ce = nn.CrossEntropyLoss()
    os.makedirs(out_dir, exist_ok=True)
    best = 0.0
    print(f"[CNN] Start: {datetime.utcnow().isoformat()}")
    for e in range(200):
        m.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad()
            o = m(x)
            loss = ce(o, y)
            loss.backward()
            opt.step()
            total += loss.item()
        m.eval()
        ok = 0
        tot = 0
        with torch.no_grad():
            for x, y in val_dl:
                o = m(x.to(dev))
                pred = o.argmax(dim=-1)
                ok += (pred == y.to(dev)).sum().item()
                tot += len(y)
        acc = ok / tot * 100
        sched.step(total)
        if acc > best:
            best = acc
            torch.save(m.state_dict(), os.path.join(out_dir, "cnn_pattern.pt"))
        if (e+1) % 20 == 0: print(f"[E{e+1}] Loss:{total/len(train_dl):.4f} Val:{acc:.1f}% Best:{best:.1f}%")
    print(f"[CNN] Done. Best:{best:.1f}%")
    return os.path.join(out_dir, "cnn_pattern.pt"), best
if __name__ == "__main__":
    db = "/content/drive/MyDrive/Forex-MM-Bot-v4.0"
    cp = os.path.join(db, "data/eurusd_real.csv")
    od = os.path.join(db, "models")
    print("="*60)
    print("CNN REAL PATTERN TRAINING")
    print(f"Data: {cp}")
    print("="*60)
    mp, acc = train(cp, od)
    print(f"Model: {mp}")
    print(f"Acc: {acc:.1f}%")
    print("="*60)
