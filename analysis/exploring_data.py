import random
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

data_transform = transforms.Compose([
    # Resize our images to 256x256
    transforms.Resize(size=(512,512)),
    # Flip the images randomly on the horizontal
    transforms.RandomHorizontalFlip(p=0.5),
    # Turn the image into a torch.Tensor
    transforms.ToTensor(),
])

def transfAndPlot():
    random.seed(42)

    final_img_pth = "../data/raiox/classified/healthy/MCUCXR_0092_0.png"

    img = Image.open(final_img_pth)
    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].imshow(img)
    ax[0].set_title(f"Original\nSize: {img.size}")
    ax[0].axis(False)

    # Transform and plot target image
    transformed_img = data_transform(img).permute(1, 2, 0) # (C, H, W) => (H, W, C)
    ax[1].imshow(transformed_img)
    ax[1].set_title(f"Transformed\nSize: {transformed_img.shape}")
    ax[1].axis("off")
    print("print img break")


transfAndPlot()