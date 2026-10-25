import cv2
import numpy as np


def preprocess_image(image, img_size=100):

    if image is None or image.size == 0:
        return None

    # cv2 reads BGR, convert to RGB
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # INTER_AREA is the right filter for shrinking
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image


def to_features(images):
    # แปลงเป็น float32 และหารด้วย 255.0 เพื่อ Normalize ข้อมูลให้อยู่ในช่วง [0.0, 1.0]
    images_array = np.array(images, dtype=np.float32) / 255.0
    return np.ascontiguousarray(images_array)