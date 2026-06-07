import os, numpy as np, torch, torch.nn as nn
class HyperNetwork(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=128, num_regimes=4, output_dim=2):
        super().__init__()
        self.regime_embed = nn.Embedding(num_regimes, hidden_dim//2)
        self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.LayerNorm(hidden_dim))
        self.hyper_head = nn.Sequential(nn.Linear(hidden_dim+hidden_dim//2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim), nn.Softmax(dim=-1))
        self.uncertainty_head = nn.Sequential(nn.Linear(hidden_dim+hidden_dim//2, hidden_dim//2), nn.ReLU(), nn.Linear(hidden_dim//2, 1), nn.Sigmoid())
    def forward(self, features, regime_id):
        enc = self.encoder(features)
        reg = self.regime_embed(regime_id)
        comb = torch.cat([enc, reg], dim=-1)
        return self.hyper_head(comb), self.uncertainty_head(comb)
class StaticFallback:
    def __init__(self):
        self.w = {"quiet":np.array([0.55,0.45]), "volatile":np.array([0.50,0.50]), "trending":np.array([0.60,0.40]), "normal":np.array([0.52,0.48])}
    def predict(self, regime, features=None):
        return self.w.get(regime, np.array([0.50,0.50])), 0.5
class DualThroatEnsemble:
    def __init__(self, config, model_path=None):
        self.dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.static = StaticFallback()
        self.use_onnx = False
        self.onnx_session = None
        if model_path and os.path.exists(model_path):
            if model_path.endswith(".onnx"): self._load_onnx(model_path)
            else: self._load_pt(model_path)
    def _load_pt(self, path):
        self.model = HyperNetwork()
        self.model.load_state_dict(torch.load(path, map_location=self.dev))
        self.model.to(self.dev)
        self.model.eval()
    def _load_onnx(self, path):
        try:
            import onnxruntime as ort
            self.onnx_session = ort.InferenceSession(path)
            self.use_onnx = True
        except: pass
    def predict(self, features, regime="normal"):
        rid = {"quiet":0, "normal":1, "volatile":2, "trending":3}.get(regime, 1)
        if self.use_onnx and self.onnx_session:
            o = self.onnx_session.run(None, {"features":features.reshape(1,-1).astype(np.float32), "regime_id":np.array([rid], dtype=np.int64)})
            w, u = o[0][0], o[1][0][0]
            src = "onnx"
        elif self.model:
            with torch.no_grad():
                ft = torch.from_numpy(features).float().unsqueeze(0).to(self.dev)
                rt = torch.tensor([rid], dtype=torch.long).to(self.dev)
                wt, ut = self.model(ft, rt)
                w, u = wt.cpu().numpy()[0], ut.cpu().numpy()[0][0]
            src = "hypernetwork"
        else:
            w, u = self.static.predict(regime, features)
            src = "static"
        return {"long_weight":float(w[0]), "short_weight":float(w[1]), "uncertainty":float(u), "regime":regime, "source":src}
    def export_onnx(self, output_path):
        if self.model is None: return
        df = torch.randn(1, 5).to(self.dev)
        dr = torch.tensor([0], dtype=torch.long).to(self.dev)
        torch.onnx.export(self.model, (df, dr), output_path, input_names=["features","regime_id"], output_names=["weights","uncertainty"], dynamic_axes={"features":{0:"batch"}, "regime_id":{0:"batch"}, "weights":{0:"batch"}, "uncertainty":{0:"batch"}}, opset_version=11)
