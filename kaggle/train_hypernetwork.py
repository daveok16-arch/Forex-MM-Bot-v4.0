import os, sys, yaml, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agents.dual_throat_ensemble import HyperNetwork
class MMDataset(Dataset):
    def __init__(self, n=50000):
        self.n = n
        np.random.seed(42)
        self.x = np.random.randn(n, 5).astype(np.float32)
        self.r = np.random.randint(0, 4, size=n)
        self.y = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            b = [0.55, 0.45] if self.r[i]==0 else [0.52, 0.48] if self.r[i]==1 else [0.50, 0.50] if self.r[i]==2 else [0.60, 0.40]
            w = np.array(b) + np.random.randn(2)*0.05
            w = np.clip(w, 0.1, 0.9)
            self.y[i] = w / w.sum()
    def __len__(self): return self.n
    def __getitem__(self, i):
        return torch.from_numpy(self.x[i]), torch.tensor(self.r[i], dtype=torch.long), torch.from_numpy(self.y[i])
def kl_loss_fn(p, t, u):
    k = torch.sum(t * torch.log(t/(p+1e-8)+1e-8), dim=-1).mean()
    e = -torch.sum(p*torch.log(p+1e-8), dim=-1).mean()*0.01
    err = torch.abs(p-t).mean(dim=-1, keepdim=True)
    ul = torch.mean((u.squeeze()-err.squeeze().detach())**2)
    return k + e + 0.1*ul
def train(cfg_path, out_dir):
    with open(cfg_path, "r") as f: cfg = yaml.safe_load(f)
    tc = cfg["training"]["hypernetwork"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] {dev}")
    m = HyperNetwork(hidden_dim=tc["hidden_dim"], num_regimes=tc["num_regimes"]).to(dev)
    print(f"[Train] Params: {sum(p.numel() for p in m.parameters()):,}")
    ds = MMDataset(50000)
    dl = DataLoader(ds, batch_size=tc["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    opt = optim.Adam(m.parameters(), lr=tc["lr"])
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=tc["epochs"])
    os.makedirs(out_dir, exist_ok=True)
    best = float("inf")
    print(f"[Train] Start: {datetime.utcnow().isoformat()}")
    for epoch in range(tc["epochs"]):
        m.train()
        total = 0.0
        for x, r, y in dl:
            x, r, y = x.to(dev), r.to(dev), y.to(dev)
            opt.zero_grad()
            pw, u = m(x, r)
            loss = kl_loss_fn(pw, y, u)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            total += loss.item()
        sched.step()
        avg = total / len(dl)
        if avg < best:
            best = avg
            torch.save(m.state_dict(), os.path.join(out_dir, "hypernetwork_v4.pt"))
        if (epoch+1) % 10 == 0: print(f"[E{epoch+1}/{tc['epochs']}] Loss:{avg:.6f} Best:{best:.6f}")
    torch.save(m.state_dict(), os.path.join(out_dir, "hypernetwork_v4.pt"))
    print(f"[Train] Saved: {os.path.join(out_dir, 'hypernetwork_v4.pt')}")
    m.eval()
    df = torch.randn(1, 5).to(dev)
    dr = torch.tensor([0], dtype=torch.long).to(dev)
    torch.onnx.export(m, (df, dr), os.path.join(out_dir, "hypernetwork_v4.onnx"), input_names=["features","regime_id"], output_names=["weights","uncertainty"], dynamic_axes={"features":{0:"batch"}, "regime_id":{0:"batch"}, "weights":{0:"batch"}, "uncertainty":{0:"batch"}}, opset_version=11)
    print(f"[Train] ONNX: {os.path.join(out_dir, 'hypernetwork_v4.onnx')}")
    vds = MMDataset(5000)
    vdl = DataLoader(vds, batch_size=256)
    ok, tot = 0, 0
    m.eval()
    with torch.no_grad():
        for x, r, y in vdl:
            pw, _ = m(x.to(dev), r.to(dev))
            for i in range(len(r)):
                if np.argmax(pw[i].cpu().numpy()) == np.argmax(y[i].numpy()): ok += 1
                tot += 1
    acc = ok/tot*100
    print(f"[Train] Val Acc: {acc:.2f}%")
    return os.path.join(out_dir, "hypernetwork_v4.pt"), os.path.join(out_dir, "hypernetwork_v4.onnx"), acc
if __name__ == "__main__":
    db = "/content/drive/MyDrive/Forex-MM-Bot-v4.0"
    cp = os.path.join(db, "config/mm_config.yaml")
    od = os.path.join(db, "models")
    print("="*60)
    print("HYPERNETWORK TRAINING")
    print(f"Start: {datetime.utcnow().isoformat()}")
    print("="*60)
    mp, op, ac = train(cp, od)
    print("="*60)
    print(f"Model: {mp}")
    print(f"ONNX: {op}")
    print(f"Acc: {ac:.2f}%")
    print("="*60)
