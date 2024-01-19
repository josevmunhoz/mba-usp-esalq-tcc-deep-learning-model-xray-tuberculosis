from torchvision import transforms

def train_transform() -> transforms:
    return transforms.Compose([
        transforms.Resize(size=(28,28)),
        transforms.Grayscale(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        ])

def test_transform() -> transforms:
    return transforms.Compose([
        transforms.Resize(size=(28,28)),
        transforms.Grayscale(),
        transforms.ToTensor(),
        ])