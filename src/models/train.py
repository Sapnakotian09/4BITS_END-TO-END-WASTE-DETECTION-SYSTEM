import torch
import torch.nn as nn
import torch.optim as optim
from src.data.preprocess import get_data_loaders
from src.models.model import get_model

DATA_DIR = "dataset"  # structure: dataset/train, dataset/val

def train():
    train_loader, val_loader, classes = get_data_loaders(DATA_DIR)

    model = get_model(len(classes))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):  # keep small for hackathon
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} done")

    torch.save(model.state_dict(), "model.pth")
    print("Model saved!")

if __name__ == "__main__":
    train()
