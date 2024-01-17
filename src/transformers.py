import torch
from torchvision import transforms

def defaultTransform() -> transforms:
    """
    Transform default for torchvision ML Models.
    Size = 64x64
    RadomHorizontalFlip = p (0.5)


    Returns:
        torchvision.transforms: Return the transformers for ML Models
    """
    return transforms.Compose([
        # Resize our images to 64x64
        transforms.Resize(size=(64,64)),
        # Flip the images randomly on the horizontal
        transforms.RandomHorizontalFlip(p=0.5),
        # Turn the image into a torch.Tensor
        transforms.ToTensor(),
        ])