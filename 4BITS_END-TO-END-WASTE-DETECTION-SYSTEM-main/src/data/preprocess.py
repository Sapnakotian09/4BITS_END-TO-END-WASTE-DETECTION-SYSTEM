import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader


def get_data_loaders(data_dir, batch_size=32):

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=transform
    )

    val_data = datasets.ImageFolder(
        os.path.join(data_dir, "val"),
        transform=transform
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_data,
        batch_size=batch_size
    )

    return train_loader, val_loader, train_data.classes
