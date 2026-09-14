"""
preprocess.py
==============
โมดูลสำหรับ "เตรียมข้อมูลภาพ" (Step 2 ของ pipeline)

มี 3 ฟังก์ชันหลัก แบ่งหน้าที่กันชัดเจน:
    1. preprocess_image()  -> แปลงภาพ "1 ภาพ" ให้เป็น grayscale ขนาดมาตรฐาน
    2. to_features()       -> แปลง array ของภาพ (n, h, w) ให้เป็น feature matrix
                               2 มิติ (n, h*w) พร้อม normalize ค่าสีให้อยู่ในช่วง 0-1
    3. preprocess_images()  -> รวมสองฟังก์ชันด้านบนเข้าด้วยกัน สำหรับใช้กับ
                               ชุดข้อมูลเล็ก ๆ ที่โหลดเป็น list ภาพดิบมาแล้ว

ทำไมต้อง normalize และ flatten ภาพ:
    - SVM (และโมเดล ML แบบดั้งเดิมทั่วไป) รับ input เป็นเวกเตอร์ตัวเลข 1 มิติ
      ต่อ 1 ตัวอย่าง ไม่ใช่ภาพ 2 มิติแบบที่ CNN ใช้ได้โดยตรง
    - การ normalize ค่าพิกเซลจาก 0-255 ให้เหลือ 0-1 ช่วยให้ scale ของ feature
      ทุกตัวใกล้เคียงกัน ทำให้โมเดลฝึกได้เสถียรขึ้น
"""

import cv2
import numpy as np


def preprocess_image(image, img_size=100):
    """
    แปลงภาพ "หนึ่งภาพ" ให้พร้อมใช้งาน:
        1) ตรวจสอบว่าภาพใช้งานได้หรือไม่ (ไม่ใช่ None/ว่างเปล่า)
        2) แปลงจากภาพสี (BGR ของ OpenCV) เป็น grayscale ถ้ายังเป็นภาพสีอยู่
        3) ย่อ/ขยายภาพให้เป็นขนาดมาตรฐาน img_size x img_size

    Args:
        image (np.ndarray | None) : ภาพที่อ่านมาจาก cv2.imread()
        img_size (int)            : ขนาดด้านที่ต้องการหลัง resize

    Returns:
        np.ndarray ภาพ grayscale ขนาด (img_size, img_size)
        หรือ None ถ้าภาพนำเข้ามาใช้งานไม่ได้ (เช่น ไฟล์เสีย)
    """

    # ป้องกัน error กรณีไฟล์เปิดไม่ได้ (cv2.imread คืน None) หรือภาพว่างเปล่า
    if image is None or image.size == 0:
        return None
    # 👉 กันโปรแกรม crash ถ้าไฟล์ภาพเสีย/เปิดไม่ได้ -> คืน None แล้วให้ data_load.py ข้ามไป

    # ภาพสีจาก OpenCV จะมี 3 มิติ (h, w, channel=BGR) -> แปลงเป็น grayscale
    # เหลือ 2 มิติ (h, w) เพื่อลดจำนวน feature ลง 3 เท่า และตัดข้อมูลสีที่ไม่จำเป็น
    # ออกไป (รูปทรง/ลวดลายสำคัญกว่าสีสำหรับงานนี้)
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 👉 ตัดข้อมูลสีทิ้ง เหลือแค่ความสว่าง (h,w) เพราะรูปทรง/ลวดลายสำคัญกว่าสีสำหรับงานนี้

    # ปรับทุกภาพให้มีขนาดเท่ากันเสมอ (จำเป็นเพราะภาพต้นฉบับมีขนาดไม่เท่ากัน
    # แต่โมเดลต้องการ feature vector ที่มีความยาวคงที่)
    # ใช้ INTER_AREA เพราะเป็นตัวกรองที่ให้ผลลัพธ์ดีที่สุดเวลา "ย่อ" ภาพ
    # (ลด aliasing เทียบกับการใช้ bilinear/nearest ตอนย่อขนาด)
    image = cv2.resize(
        image,
        (img_size, img_size),
        interpolation=cv2.INTER_AREA
    )

    return image
    # 👉 ทุกภาพต้องขนาดเท่ากันหมด เพราะต่อไปจะ flatten เป็นเวกเตอร์ที่ต้องยาวเท่ากันทุกภาพ


def to_features(images):
    """
    แปลง array ภาพ grayscale หลายภาพ (n, h, w) ให้เป็น feature matrix
    2 มิติ (n, h*w) ที่ SVM ใช้ฝึก/ทำนายได้โดยตรง

    ขั้นตอน:
        1) reshape ภาพแต่ละภาพจาก 2 มิติ (h, w) ให้แบน (flatten) เป็นเวกเตอร์
           1 มิติยาว h*w (เช่น 100x100 -> เวกเตอร์ 10,000 ค่า)
        2) แปลงชนิดข้อมูลเป็น float32 (จาก uint8 เดิม)
        3) normalize ค่าพิกเซลจากช่วง 0-255 ให้อยู่ในช่วง 0-1

    Args:
        images (np.ndarray) : shape (n, h, w) ค่าพิกเซล uint8 (0-255)

    Returns:
        np.ndarray shape (n, h*w) ค่า float32 อยู่ในช่วง 0-1
    """

    # -1 บอกให้ numpy คำนวณขนาดมิติที่สองให้เอง (เท่ากับ h*w)
    features = images.reshape(len(images), -1).astype(np.float32)
    # Normalize: หารด้วย 255 เพื่อให้ค่าพิกเซลอยู่ในช่วง [0, 1] แทน [0, 255]
    features /= 255.0

    return features
    # 👉 ภาพ 100x100 หนึ่งภาพ กลายเป็นเวกเตอร์ตัวเลข 10,000 ค่า อยู่ในช่วง 0-1 พร้อมป้อนเข้า SVM


def preprocess_images(images, img_size=100):
    """
    ฟังก์ชันรวม (convenience function) สำหรับ "ภาพดิบที่โหลดมาเป็น list แล้ว"
    ใช้ preprocess_image() กับทุกภาพ กรองภาพที่ใช้งานไม่ได้ (None) ออก
    แล้วส่งต่อให้ to_features() แปลงเป็น feature matrix

    หมายเหตุ: เหมาะกับชุดข้อมูลขนาดเล็กที่โหลดภาพทั้งหมดไว้ใน list ก่อนแล้ว
    ต่างจาก data_load.py ที่แปลงภาพทีละภาพระหว่างอ่านไฟล์เพื่อประหยัด RAM
    กับชุดข้อมูลขนาดใหญ่

    Args:
        images (list) : list ของภาพดิบ (ผลจาก cv2.imread แต่ละไฟล์)
        img_size (int): ขนาดภาพหลัง resize

    Returns:
        np.ndarray feature matrix shape (n, img_size*img_size) ค่า 0-1
    """

    processed = [preprocess_image(img, img_size) for img in images]
    # กรองภาพที่แปลงไม่สำเร็จ (คืนค่า None) ออกก่อนรวมเป็น array เดียว
    processed = [img for img in processed if img is not None]

    return to_features(np.stack(processed))
