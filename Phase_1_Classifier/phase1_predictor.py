import torch
import torch.nn as nn
from img2vec_pytorch import Img2Vec
from utils.image_utils import extract_features, apply_clahe

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import numpy as np
import cv2
import base64
from io import BytesIO


class PneumoniaPredictor:
    def __init__(self, model_path, device=None, threshold=0.4):
        self.device = device if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.img2vec = Img2Vec(
            model="densenet121",
            cuda=torch.cuda.is_available()
        )

        self.threshold = threshold
        self.model = self._build_model()
        self._load_weights(model_path)

        # backbone for gradcam
        self.backbone = self.img2vec.model
        self.target_layers = [self.backbone.features[-1]]

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
            "confidence": round(confidence * 100, 2)
        }

    def generate_gradcam(self, image):
        img = apply_clahe(image)
        img = np.array(img.resize((244, 244))).astype(np.float32) / 255.0

        input_tensor = (
            torch.tensor(img)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .float()
            .to(self.device)
        )

        cam = GradCAM(
            model=self.backbone,
            target_layers=self.target_layers
        )

        grayscale_cam = cam(input_tensor=input_tensor)[0]

        visualization = show_cam_on_image(
            img,
            grayscale_cam,
            use_rgb=True,
            image_weight=0.7
        )

        _, buffer = cv2.imencode(".jpg", visualization)
        encoded = base64.b64encode(buffer).decode("utf-8")

        return encoded