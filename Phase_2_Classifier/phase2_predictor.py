# phase2_predictor.py
import torch
import torch.nn as nn
from img2vec_pytorch import Img2Vec
from utils.image_utils import extract_features

class BacterialViralPredictor:
    def __init__(self, model_path, device=None, threshold=0.53):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.img2vec = Img2Vec(model="densenet121", cuda=torch.cuda.is_available())
        self.threshold = threshold
        self.model = self._build_model()
        self._load_weights(model_path)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(1024, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),

            nn.Linear(64, 1)
        )
        return model.to(self.device)

    def _load_weights(self, model_path):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, image):
        vector = extract_features(self.img2vec, image)
        vector = torch.tensor(vector).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            prob = torch.sigmoid(self.model(vector)).item()
        return {
            "label": "VIRAL" if prob > self.threshold else "BACTERIAL",
            "confidence": round(prob * 100, 2)
        }