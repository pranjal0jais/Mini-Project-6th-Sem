from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

def validate(model, loss_function, threshold, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)

            output = model(x)
            loss = loss_function(output, y)

            total_loss += loss.item() * x.size(0)

            probs = torch.sigmoid(output)
            preds = (probs > threshold).float()

            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"Valid - Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}")
    return avg_loss

def train(model, loss_function, threshold, optimizer, train_loader, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device).view(-1, 1)

        optimizer.zero_grad()

        output = model(x)
        loss = loss_function(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)

        probs = torch.sigmoid(output)
        preds = (probs > threshold).float()

        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total

    print(f"Train - Loss: {avg_loss:.4f} Accuracy: {accuracy:.4f}")

def evaluate(model, threshold, data_loader, device, labels):
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)
            y = y.to(device).view(-1, 1)

            outputs = model(x)
            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds)
    recall = recall_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    print("===== EVALUATION METRICS =====")
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")
    print("\nConfusion Matrix:\n", cm)

    print("\nDetailed Classification Report:\n")
    print(classification_report(all_targets, all_preds))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=labels)
    disp.plot(cmap="Blues")
    plt.grid(False)
    plt.title("Confusion Matrix")
    plt.show()

def get_tensor_dataset(dataframe, label):
    feat = dataframe.drop(label, axis=1).values
    label = dataframe[label].values
    dataset = TensorDataset(
        torch.tensor(feat, dtype=torch.float32),
        torch.tensor(label, dtype=torch.float32)
    )
    return dataset

def get_data_loader(dataset, batch_size=32):
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True)
    return loader
