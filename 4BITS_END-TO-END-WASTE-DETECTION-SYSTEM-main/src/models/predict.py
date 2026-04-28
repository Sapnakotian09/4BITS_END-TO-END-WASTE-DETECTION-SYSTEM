import torch

from PIL import Image

from torchvision import transforms

from src.models.model import get_model


CLASSES = [
    "plastic",
    "metal",
    "paper",
    "glass"
]


def predict_image(image_path):

    model = get_model(len(CLASSES))

    model.load_state_dict(
        torch.load("model.pth")
    )

    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path)

    image = transform(image).unsqueeze(0)

    output = model(image)

    _, predicted = torch.max(output, 1)

    predicted_class = CLASSES[predicted.item()]

    return predicted_class


if __name__ == "__main__":

    image_path = "sample.jpg"

    prediction = predict_image(image_path)

    print(f"Predicted Waste Type: {prediction}")
