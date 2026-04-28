import torch
import torch.nn as nn
import torch.optim as optim

from src.data.preprocess import get_data_loaders
from src.models.model import get_model


DATA_DIR = "dataset"


def train():

    train_loader, val_loader, classes = get_data_loaders(DATA_DIR)

    model = get_model(len(classes))

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001
    )

    epochs = 3

    for epoch in range(epochs):

        model.train()

        running_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs} completed")

    torch.save(model.state_dict(), "model.pth")

    print("Model saved successfully")


if __name__ == "__main__":
    train()
