"""
split_data.py
===============
โมดูลสำหรับ "แบ่งชุดข้อมูล" (Step 3 ของ pipeline)

หน้าที่: แบ่งข้อมูลทั้งหมด (X, y) ออกเป็นชุด Train และชุด Test
เพื่อให้สามารถประเมินว่าโมเดลทำงานได้ดีแค่ไหนกับข้อมูลที่ "ไม่เคยเห็นมาก่อน"
(ถ้าใช้ข้อมูลเดียวกับตอนฝึกมาวัดผล จะได้ค่า accuracy ที่สูงเกินจริง)

จุดสำคัญของการแบ่งข้อมูลในไฟล์นี้:
    - stratify=y  : รักษาสัดส่วนของแต่ละคลาสให้เท่ากันทั้งในชุด train และ test
                    เช่น ถ้าข้อมูลจริงมี banana:orange = 50:50 ชุด train/test
                    ที่ได้ก็จะยังคงสัดส่วนประมาณ 50:50 เช่นกัน ไม่ใช่สุ่มจนคลาส
                    ใดคลาสหนึ่งขาดหายไปในชุด test
    - random_state=42 : fix ค่าการสุ่มให้ผลลัพธ์การแบ่งข้อมูลเหมือนเดิมทุกครั้ง
                    ที่รันโค้ด (reproducible) เพื่อให้เปรียบเทียบผลการทดลองได้ตรงกัน
"""

import numpy as np
from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2):
    """
    แบ่งข้อมูล feature (X) และ label (y) เป็นชุด train/test แบบ stratified

    Args:
        X (array-like)   : feature matrix ทั้งหมด shape (n_samples, n_features)
        y (array-like)   : label ทั้งหมด shape (n_samples,)
        test_size (float): สัดส่วนข้อมูลที่แบ่งไปเป็นชุด test (0.2 = 20%)

    Returns:
        X_train, X_test, y_train, y_test : ข้อมูลที่แบ่งแล้วตามสัดส่วนที่กำหนด
    """
    # sklearn ต้องการให้ y เป็น numpy array (ไม่ใช่ list ธรรมดา) เพื่อให้
    # พารามิเตอร์ stratify ใช้งานได้ถูกต้อง
    y = np.asarray(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=42,   # กำหนด seed ให้ผลการสุ่มแบ่งข้อมูลคงที่ ทำซ้ำได้
        stratify=y          # รักษาสัดส่วนคลาสให้เท่ากันทั้งชุด train และ test
    )
    # 👉 เช่น test_size=0.2 -> ข้อมูล 100 ภาพ จะได้ train 80 / test 20 ภาพ โดยสัดส่วนคลาสยังเท่าเดิม

    return X_train, X_test, y_train, y_test
