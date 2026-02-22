import torch

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
