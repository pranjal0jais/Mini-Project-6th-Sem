import torch

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
