import albumentations as A
from PIL import Image
import numpy as np


def augment_image(img: Image.Image):
    img_np = np.array(img)

    transform = A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(p=0.5),
        A.HueSaturationValue(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.7),
        A.RandomShadow(p=0.3),
    ])

    for i in range(5):
        augmented = transform(image=img_np)
        pil_img = Image.fromarray(augmented['image'])
