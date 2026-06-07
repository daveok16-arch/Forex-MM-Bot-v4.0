import os, numpy as np
import onnxruntime as ort
class StaticFallback:
    def __init__(self):
        self.w = {"quiet":np.array([0.55,0.45]), "volatile":np.array([0.50,0.50]), "trending":np.array([0.60,0.40]), "normal":np.array([0.52,0.48])}
    def predict(self, regime, features=None):
        return self.w.get(regime, np.array([0.50,0.50])), 0.5
class DualThroatEnsemble:
    def __init__(self, config, model_path=None):
        self.static = StaticFallback()
        self.use_onnx = False
        self.onnx_session = None
        if model_path and os.path.exists(model_path):
            if model_path.endswith(".onnx"):
                try:
                    self.onnx_session = ort.InferenceSession(model_path)
                    self.use_onnx = True
                    print(f"[Ensemble] ONNX loaded: {model_path}")
                except Exception as e:
                    print(f"[Ensemble] ONNX error: {e}")
            else:
                print(f"[Ensemble] PT found but no PyTorch: {model_path}")
    def predict(self, features, regime="normal"):
        rid = {"quiet":0, "normal":1, "volatile":2, "trending":3}.get(regime, 1)
        if self.use_onnx and self.onnx_session:
            o = self.onnx_session.run(None, {"features":features.reshape(1,-1).astype(np.float32), "regime_id":np.array([rid], dtype=np.int64)})
            w, u = o[0][0], o[1][0][0]
            src = "onnx"
        else:
            w, u = self.static.predict(regime, features)
            src = "static"
        return {"long_weight":float(w[0]), "short_weight":float(w[1]), "uncertainty":float(u), "regime":regime, "source":src}
