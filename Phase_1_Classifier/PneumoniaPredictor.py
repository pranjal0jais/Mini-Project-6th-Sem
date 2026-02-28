import torch
import torch.nn as nn
from img2vec_pytorch import Img2Vec

from utils.image_utils import extract_features


class PneumoniaPredictor:
    def __init__(self, model_path, device=None, threshold=0.4):
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.img2vec = Img2Vec(model="densenet121", cuda=True)
        self.threshold = threshold
        self.model = self._build_model()
        self._load_weights(model_path)

    def _build_model(self):
        model = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )
        return model.to(self.device)

    def _load_weights(self, model_path):
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device)
        )
        self.model.eval()

    def predict(self, image):
        vector = extract_features(self.img2vec, image)
        vector = torch.tensor(vector)
        if vector.dim() == 1:
            vector = vector.unsqueeze(0)

        vector = vector.float().to(self.device)

        with torch.no_grad():
            output = self.model(vector)
            prob = torch.sigmoid(output)
            pred = (prob > self.threshold).float()

        confidence = prob.item()
        label = "PNEUMONIA" if pred.item() == 1 else "NORMAL"

        return {
            "label": label,
            "confidence": round(confidence, 4)*100
        }