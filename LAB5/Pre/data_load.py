"""
data_load.py
=============
โมดูลสำหรับ "โหลดข้อมูลดิบ" (Step 1 ของ pipeline)

หน้าที่หลัก:
    - สแกนโฟลเดอร์ data set เพื่อหาคลาสทั้งหมดโดยอัตโนมัติ
      (แต่ละโฟลเดอร์ย่อย = 1 คลาส เช่น freshbanana, freshoranges)
    - อ่านไฟล์ภาพทีละไฟล์ด้วย OpenCV แล้วส่งต่อให้ preprocess_image()
      แปลงเป็น grayscale + resize ทันที เพื่อไม่ให้ต้องเก็บภาพขนาดเต็ม
      ทุกภาพไว้ใน memory พร้อมกัน (ประหยัด RAM เวลาข้อมูลมีหลายพันภาพ)
    - ข้ามไฟล์ที่เสีย/เปิดไม่ได้โดยอัตโนมัติ พร้อมนับจำนวนที่ข้ามไปรายงาน
    - จำกัดจำนวนภาพต่อคลาสได้ด้วย max_per_class (ควบคุมเวลาฝึกโมเดล)

ผลลัพธ์ที่ได้จากฟังก์ชัน load_data():
    images  : numpy array รูปภาพ grayscale ขนาด (n, img_size, img_size)
    labels  : numpy array ตัวเลข label ของแต่ละภาพ (0, 1, ... ตามลำดับคลาส)
    classes : list ชื่อคลาสเรียงตามลำดับ index ที่ใช้เป็น label
"""

import os
import cv2
import numpy as np

from preprocess import preprocess_image

# นามสกุลไฟล์ภาพที่ยอมรับ ไฟล์อื่นที่ไม่ใช่นามสกุลเหล่านี้จะถูกข้ามไปเลย
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp")


def load_data(data_path, img_size=100, max_per_class=None):
    """
    โหลดภาพทั้งหมดจากโครงสร้างโฟลเดอร์แบบ:
        data_path/
            classA/  -> ภาพของคลาส A
            classB/  -> ภาพของคลาส B
            ...

    Args:
        data_path (str)      : path ไปยังโฟลเดอร์ dataset หลัก
        img_size (int)       : ขนาดภาพหลัง resize (เป็นสี่เหลี่ยมจัตุรัส img_size x img_size)
        max_per_class (int)  : จำนวนภาพสูงสุดที่จะโหลดต่อ 1 คลาส
                                (None = โหลดทั้งหมด แต่จะช้ามากถ้าข้อมูลเยอะ)

    Returns:
        images  (np.ndarray) : shape (n_samples, img_size, img_size) แบบ grayscale
        labels  (np.ndarray) : shape (n_samples,) label เป็นเลขจำนวนเต็ม 0..n_classes-1
        classes (list[str])  : ชื่อคลาสเรียงตาม index ของ label
    """

    images = []
    labels = []

    # ---- ตรวจจับคลาสอัตโนมัติจากชื่อโฟลเดอร์ย่อย ----
    # เรียงชื่อ (sorted) เพื่อให้ label ของแต่ละคลาสคงที่เหมือนเดิมทุกครั้งที่รัน
    classes = sorted([
        folder
        for folder in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, folder))
    ])
    print("Detected classes:", classes)
    # 👉 เช่น มีโฟลเดอร์ cat/, dog/ -> classes = ["cat", "dog"], cat=label 0, dog=label 1

    # ---- วนอ่านภาพทีละคลาส ----
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_path, class_name)
        # เรียงชื่อไฟล์เพื่อให้ลำดับการโหลดคงที่ (reproducible) และกรองเฉพาะ
        # ไฟล์ที่นามสกุลอยู่ใน VALID_EXT เท่านั้น
        filenames = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(VALID_EXT)
        )

        loaded = 0
        skipped = 0
        for filename in filenames:
            # หยุดโหลดคลาสนี้ทันทีเมื่อครบโควตา max_per_class
            if max_per_class and loaded >= max_per_class:
                break

            image_path = os.path.join(class_path, filename)
            image = cv2.imread(image_path)  # อ่านไฟล์ภาพ (คืนค่า None ถ้าเปิดไม่ได้)

            # แปลง grayscale + resize ทันทีที่นี่ (ไม่ใช่ตอนหลัง) เพื่อไม่ต้อง
            # เก็บภาพสีขนาดเต็มของทุกไฟล์ไว้ใน RAM พร้อมกันทั้งหมด
            image = preprocess_image(image, img_size)
            # 👉 แปลง grayscale+resize ทันทีทีละภาพ ไม่รอสะสมทุกภาพก่อน เพื่อประหยัด RAM

            # ไฟล์เสีย/เปิดไม่ได้/ภาพว่าง -> preprocess_image คืนค่า None -> ข้ามไป
            if image is None:
                skipped += 1
                continue

            images.append(image)
            labels.append(label)
            loaded += 1

        print(f"Loaded class {class_name}: {loaded} images ({skipped} skipped)")
        # 👉 skipped คือไฟล์เสีย/เปิดไม่ได้ ถูกข้ามไปเฉยๆ ไม่ทำให้ทั้งโปรแกรม error

    # รวมภาพทั้งหมดเป็น numpy array เดียว (stack ตาม axis แรก)
    return np.stack(images), np.array(labels), classes
    # 👉 np.stack รวม list ของภาพ (h,w) หลายๆ ภาพ ให้เป็น array เดียว shape (n,h,w)
