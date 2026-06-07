import os, sys, yaml, numpy as np, torch, torch.nn as nn, torch.optim as optim
from torch.distributions import Categorical
from datetime import datetime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
class MMEnv:
    def __init__(self, n=1000):
        self.n = n
        self.step_n = 0
        self.price = 1.1000
        self.inv = 0.0
        self.pnl = 0.0
        self.max_inv = 10.0
    def reset(self):
        self.step_n = 0
        self.price = 1.1000 + np.random.randn()*0.001
        self.inv = 0.0
        self.pnl = 0.0
        return self._obs()
    def _obs(self):
        return np.array([self.price, self.inv/self.max_inv, 0.001+abs(np.random.randn())*0.0005, abs(np.random.randn())*0.001, 1.0-self.step_n/self.n], dtype=np.float32)
    def step(self, a):
        self.step_n += 1
        self.price += np.random.randn()*0.0002
        sp = 0.001
        fp = 0.0
        if a == 0: sp = 0.0005; fp = 0.8
        elif a == 1: sp = 0.002; fp = 0.3
        elif a == 2: sp = 0.001; fp = 0.6
        elif a == 3: sp = 0.001; fp = 0.6
        elif a == 4: self.pnl -= abs(self.inv)*0.0001; self.inv = 0.0; fp = 0.0
        if a in [0,1] and np.random.rand() < fp:
            s = 1 if np.random.rand() < 0.5 else -1
            self.inv += s*1.0
            self.pnl += sp/2
        if a == 2 and np.random.rand() < fp:
            self.inv -= 1.0
            self.pnl += sp/2
        if a == 3 and np.random.rand() < fp:
            self.inv += 1.0
            self.pnl += sp/2
        ip = (self.inv/self.max_inv)**2 * 0.01
        sr = (0.002-sp)*10 if a in [0,1] else 0.0
        r = sr - ip + self.pnl*0.001
        done = self.step_n >= self.n
        trunc = abs(self.inv) >= self.max_inv
        return self._obs(), r, done or trunc, {"pnl": self.pnl}
class ActorCritic(nn.Module):
    def __init__(self, obs=5, act=5, hid=256):
        super().__init__()
        self.s = nn.Sequential(nn.Linear(obs, hid), nn.ReLU(), nn.LayerNorm(hid), nn.Linear(hid, hid), nn.ReLU(), nn.LayerNorm(hid))
        self.a = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Linear(hid//2, act))
        self.c = nn.Sequential(nn.Linear(hid, hid//2), nn.ReLU(), nn.Linear(hid//2, 1))
    def forward(self, o):
        s = self.s(o)
        return self.a(s), self.c(s)
    def get_action(self, o, det=False):
        l, v = self.forward(o)
        d = Categorical(logits=l)
        a = d.probs.argmax(dim=-1) if det else d.sample()
        return a, d.log_prob(a), v
class Buffer:
    def __init__(self, cap, obs):
        self.o = np.zeros((cap, obs), dtype=np.float32)
        self.a = np.zeros(cap, dtype=np.int64)
        self.r = np.zeros(cap, dtype=np.float32)
        self.v = np.zeros(cap, dtype=np.float32)
        self.lp = np.zeros(cap, dtype=np.float32)
        self.d = np.zeros(cap, dtype=np.float32)
        self.ptr = 0
        self.cap = cap
    def store(self, o, a, r, v, lp, d):
        i = self.ptr % self.cap
        self.o[i] = o
        self.a[i] = a
        self.r[i] = r
        self.v[i] = v
        self.lp[i] = lp
        self.d[i] = float(d)
        self.ptr += 1
    def get(self):
        return (torch.from_numpy(self.o[:self.ptr]), torch.from_numpy(self.a[:self.ptr]), torch.from_numpy(self.r[:self.ptr]), torch.from_numpy(self.v[:self.ptr]), torch.from_numpy(self.lp[:self.ptr]), torch.from_numpy(self.d[:self.ptr]))
    def clear(self): self.ptr = 0
def gae(r, v, d, g=0.99, l=0.95):
    adv = np.zeros_like(r)
    la = 0
    for t in reversed(range(len(r))):
        nv = v[t+1] if t < len(r)-1 else 0
        delta = r[t] + g*nv*(1-d[t]) - v[t]
        adv[t] = la = delta + g*l*(1-d[t])*la
    return adv, adv + v
def train(cfg_path, out_dir):
    with open(cfg_path, "r") as f: cfg = yaml.safe_load(f)
    rc = cfg["training"]["rl"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[RL] {dev}")
    m = ActorCritic().to(dev)
    opt = optim.Adam([{"params": m.a.parameters(), "lr": rc["lr_actor"]}, {"params": m.c.parameters(), "lr": rc["lr_critic"]}])
    print(f"[RL] Params: {sum(p.numel() for p in m.parameters()):,}")
    env = MMEnv(1000)
    buf = Buffer(rc["steps_per_epoch"], 5)
    os.makedirs(out_dir, exist_ok=True)
    best = -float("inf")
    print(f"[RL] Start: {datetime.utcnow().isoformat()}")
    for epoch in range(rc["ppo_epochs"]):
        o = env.reset()
        er = 0.0
        ep = 0.0
        for step in range(rc["steps_per_epoch"]):
            ot = torch.from_numpy(o).float().unsqueeze(0).to(dev)
            with torch.no_grad():
                a, lp, v = m.get_action(ot)
            an = a.cpu().numpy()[0]
            lpn = lp.cpu().numpy()[0]
            vn = v.cpu().numpy()[0][0]
            no, r, done, info = env.step(an)
            buf.store(o, an, r, vn, lpn, done)
            er += r
            ep = info["pnl"]
            o = no
            if done: o = env.reset()
        obs, act, rew, val, logp, don = buf.get()
        obs, act, rew, val, logp, don = obs.to(dev), act.to(dev), rew.to(dev), val.to(dev), logp.to(dev), don.to(dev)
        adv, ret = gae(rew.cpu().numpy(), val.cpu().numpy(), don.cpu().numpy(), rc["gamma"], rc["lambda_gae"])
        adv = torch.from_numpy(adv).float().to(dev)
        ret = torch.from_numpy(ret).float().to(dev)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        for _ in range(10):
            l, v = m(obs)
            d = Categorical(logits=l)
            nlp = d.log_prob(act)
            ent = d.entropy().mean()
            ratio = torch.exp(nlp - logp)
            s1 = ratio * adv
            s2 = torch.clamp(ratio, 1-rc["clip_ratio"], 1+rc["clip_ratio"]) * adv
            al = -torch.min(s1, s2).mean()
            cl = nn.MSELoss()(v.squeeze(), ret)
            loss = al + 0.5*cl - 0.01*ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 0.5)
            opt.step()
        buf.clear()
        if (epoch+1) % 50 == 0: print(f"[E{epoch+1}/{rc['ppo_epochs']}] Reward:{er:.4f} PnL:{ep:.4f}")
        if er > best:
            best = er
            torch.save(m.state_dict(), os.path.join(out_dir, "agent_f_rl_v4.pt"))
    torch.save(m.state_dict(), os.path.join(out_dir, "agent_f_rl_v4.pt"))
    print(f"[RL] Saved: {os.path.join(out_dir, 'agent_f_rl_v4.pt')}")
    m.eval()
    do = torch.randn(1, 5).to(dev)
    torch.onnx.export(m, do, os.path.join(out_dir, "agent_f_rl_v4.onnx"), input_names=["obs"], output_names=["logits","value"], dynamic_axes={"obs":{0:"batch"}, "logits":{0:"batch"}, "value":{0:"batch"}}, opset_version=11)
    print(f"[RL] ONNX: {os.path.join(out_dir, 'agent_f_rl_v4.onnx')}")
    return os.path.join(out_dir, "agent_f_rl_v4.pt"), os.path.join(out_dir, "agent_f_rl_v4.onnx"), best
if __name__ == "__main__":
    db = "/content/drive/MyDrive/Forex-MM-Bot-v4.0"
    cp = os.path.join(db, "config/mm_config.yaml")
    od = os.path.join(db, "models")
    print("="*60)
    print("RL AGENT F TRAINING (PPO)")
    print(f"Start: {datetime.utcnow().isoformat()}")
    print("="*60)
    mp, op, br = train(cp, od)
    print("="*60)
    print(f"Model: {mp}")
    print(f"ONNX: {op}")
    print(f"Best Reward: {br:.4f}")
    print("="*60)
