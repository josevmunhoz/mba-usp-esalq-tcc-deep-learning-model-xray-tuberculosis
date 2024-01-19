from torchvision import datasets

def defaultImgLoader(path, transform) -> datasets.ImageFolder:
    """Defatult loader for images

    Args:
        path (List[String]): Path were the img are stores already classified
        transform (torchvision.utils.transforms): Transformers functions with/without Composer

    Returns:
        datasets.ImageFolder: data ready to be trained or tested.
    """
    return datasets.ImageFolder(root=path,
                                transform=transform,
                                target_transform=None)