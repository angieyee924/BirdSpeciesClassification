from torchvision import transforms


def get_train_transform():
    """
    Transformations used during training.

    Random augmentation helps reduce overfitting and improves
    the model's ability to generalise to unseen images.
    """

    return transforms.Compose([
        transforms.RandomResizedCrop(
            224,
            scale=(0.7, 1.0)
        ),

        transforms.RandomHorizontalFlip(),

        transforms.RandomRotation(15),

        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2
        ),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def get_val_transform():
    """
    Deterministic preprocessing used for validation/testing.

    Random augmentation is deliberately not used here because
    evaluation should remain consistent.
    """

    return transforms.Compose([
        transforms.Resize(256),

        transforms.CenterCrop(224),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])