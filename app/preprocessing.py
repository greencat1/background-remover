from PIL import Image
from torchvision import transforms


transform = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


def preprocess(image):

    image = image.convert("RGB")

    tensor = transform(image)

    return tensor.unsqueeze(0)