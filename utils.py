from PIL import Image
from img2vec_pytorch import Img2Vec
import torch
import cv2
import numpy as np

img2vec = Img2Vec(model="densenet121", cuda=True)

def apply_clahe(img):
    img_arr = np.array(img.convert('L'))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced_img = clahe.apply(img_arr)
    return Image.fromarray(enhanced_img).convert("RGB")

def extract_features(path):
    img = Image.open(path).convert('RGB')
    img = apply_clahe(img)
    return img2vec.get_vec(img, tensor=False)

def validate(model, loss_function, threshold,val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            loss = loss_function(output, y)
            total_loss += loss.item()

            probs = torch.sigmoid(output)
            preds = (probs > threshold).float()
            correct += (preds == y).sum().item()
            total += y.size(0)

    print(f"Valid - Loss: {total_loss:.4f} Accuracy: {correct/total:.4f}")
    return total_loss


def train(model, loss_function, threshold, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        output = model(x)
        loss = loss_function(output, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        probs = torch.sigmoid(output)
        preds = (probs > threshold).float()
        correct += (preds == y).sum().item()
        total += y.size(0)

    print(f"Train - Loss: {total_loss:.4f} Accuracy: {correct/total:.4f}")
