"""
main.py
========
ไฟล์ orchestrator หลักของระบบ ทำหน้าที่เรียกใช้ทุกโมดูลตามลำดับขั้นตอน
ทั้งหมดของ pipeline (เทียบกับ flowchart ในสไลด์: เริ่มต้น -> โหลดข้อมูล ->
เตรียม/แบ่งข้อมูล -> ฝึก/ทำนาย -> บันทึกผล -> สิ้นสุด)

ลำดับการทำงาน:
    1) โหลดข้อมูลภาพดิบ + label     (data_load.py)
    2) แปลงภาพเป็น feature vector    (preprocess.py)
    3) แบ่งข้อมูล train / test       (split_data.py)
    4) ฝึกโมเดล SVM                  (svm_model.py)
    5) ทำนายผลบนชุด test             (svm_model.py)
    6) ประเมินผล + บันทึก confusion matrix (evaluate.py)

ทุกขั้นตอนจะบันทึกผลลัพธ์ระหว่างทาง (ภาพ, label, โมเดล, ชุดข้อมูลที่แบ่งแล้ว)
ลงในโฟลเดอร์ outputs/ เพื่อให้ไฟล์อื่น เช่น test_svm.py นำกลับมาใช้ซ้ำได้
โดยไม่ต้องรัน pipeline ทั้งหมดใหม่ทุกครั้ง
"""

import json
import os

import joblib
import numpy as np

from data_load import load_data
from preprocess import to_features
from split_data import split_dataset
from svm_model import train_svm, predict_svm
from evaluate import evaluate_model
# 👉 ไฟล์นี้ไม่มี logic ของตัวเอง เป็นแค่ตัวเรียกใช้ฟังก์ชันจาก 5 โมดูลตามลำดับ

# ---- ค่าคงที่ที่ตั้งค่าได้ (config) ----
DATA_PATH = "data set"      # โฟลเดอร์ dataset ต้นทาง (มีโฟลเดอร์ย่อยตามคลาส)
OUTPUT_DIR = "outputs"      # โฟลเดอร์ปลายทางสำหรับเก็บผลลัพธ์ทุกอย่าง
IMG_SIZE = 100               # ขนาดภาพมาตรฐานหลัง resize
TEST_SIZE = 0.2              # สัดส่วนข้อมูลที่แบ่งไปเป็นชุด test
MAX_PER_CLASS = 3000         # จำกัดจำนวนภาพต่อคลาส (None = ใช้ทั้งหมด แต่จะช้ามาก)


def main():

    print("--" * 30)
    print("SVM Image Recognition: Cat vs Dog")
    print("--" * 30)

    # สร้างโฟลเดอร์ outputs ถ้ายังไม่มี (exist_ok=True กันไม่ให้ error ถ้ามีอยู่แล้ว)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========== Step 1: โหลดข้อมูล ==========
    print("\n[Step 1] Loading dataset...")
    images, labels, classes = load_data(DATA_PATH, IMG_SIZE, MAX_PER_CLASS)

    # บันทึกข้อมูลดิบไว้เผื่อใช้ตรวจสอบ/รันซ้ำภายหลังโดยไม่ต้องโหลดไฟล์ภาพใหม่
    np.save(f"{OUTPUT_DIR}/images.npy", images)
    np.save(f"{OUTPUT_DIR}/labels.npy", labels)
    with open(f"{OUTPUT_DIR}/classes.json", "w") as f:
        json.dump(classes, f)

    print("\nDataset loaded successfully.")
    print(f"Total images : {len(images)}")
    print(f"Classes      : {classes}")
    # 👉 ได้ภาพดิบ (แปลง grayscale+resize แล้ว) กับ label ตัวเลข พร้อมส่งต่อไป preprocess

    # ========== Step 2: เตรียมข้อมูล (Preprocessing) ==========
    print("\n[Step 2] preprocess images...")

    # แปลงภาพ grayscale 2 มิติ (n, h, w) ให้เป็น feature matrix 2 มิติ
    # (n, h*w) พร้อม normalize ค่าพิกเซลให้อยู่ในช่วง 0-1
    X = to_features(images)
    y = labels
    print(f"Feature shape: {X.shape}")
    # 👉 ได้ X = ตารางตัวเลข (n ภาพ x h*w พิกเซล) ที่ SVM เอาไปใช้ฝึกได้โดยตรง

    # ========== Step 3: แบ่งชุดข้อมูล Train/Test ==========
    print("\n[Step 3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_dataset(X, y, TEST_SIZE)

    # บันทึกชุดข้อมูลที่แบ่งแล้ว เพื่อให้ test_svm.py นำ X_test/y_test
    # กลับมาใช้สุ่มทดสอบภาพได้โดยไม่ต้องรัน pipeline ใหม่ทั้งหมด
    np.save(f"{OUTPUT_DIR}/X_train.npy", X_train)
    np.save(f"{OUTPUT_DIR}/X_test.npy", X_test)
    np.save(f"{OUTPUT_DIR}/y_train.npy", y_train)
    np.save(f"{OUTPUT_DIR}/y_test.npy", y_test)

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples : {len(X_test)}")
    # 👉 แยกไว้ 2 กอง: train เอาไว้สอนโมเดล, test เก็บไว้วัดผลแบบไม่ให้โมเดลเคยเห็นมาก่อน

    # ========== Step 4: ฝึกโมเดล SVM ==========
    print("\n[Step 4] Training SVM...")

    # train_svm คืนค่าทั้งโมเดล SVC และ scaler (Pipeline ของ
    # StandardScaler+PCA) ที่ fit กับข้อมูล train เรียบร้อยแล้ว
    model, scaler = train_svm(X_train, y_train)

    # บันทึกโมเดลและ scaler ด้วย joblib (เหมาะกับ object ของ sklearn)
    # เพื่อให้นำกลับมาใช้ทำนายภาพใหม่ได้ทันทีโดยไม่ต้องฝึกใหม่
    joblib.dump(model, f"{OUTPUT_DIR}/svm_model.pkl")
    joblib.dump(scaler, f"{OUTPUT_DIR}/scaler.pkl")

    print("SVM training completed.")
    # 👉 ได้โมเดล SVM ที่ฝึกเสร็จแล้ว + scaler (Scaler+PCA) ที่ fit กับข้อมูล train ไว้ใช้ซ้ำตอนทำนาย

    # ========== Step 5: ทำนายผลบนชุด Test ==========
    print("\n[Step 5] Testing model...")
    predictions = predict_svm(model, scaler, X_test)
    # 👉 ได้ label ที่โมเดล "เดา" สำหรับภาพในชุด test ทุกภาพ ยังไม่รู้ว่าถูกหรือผิด

    # ========== Step 6: ประเมินผล ==========
    print("\n[Step 6] Evaluating model...")
    # คำนวณ accuracy, classification report และวาด/บันทึก confusion matrix
    # เป็นไฟล์ภาพไว้ที่ outputs/confusion_matrix.png
    evaluate_model(y_test, predictions, classes,
                   save_path=f"{OUTPUT_DIR}/confusion_matrix.png")
    # 👉 เทียบ predictions กับ y_test (คำตอบจริง) เพื่อสรุปว่าโมเดลแม่นแค่ไหน จบ pipeline


if __name__ == "__main__":
    # เงื่อนไขนี้ทำให้ main() ถูกเรียกก็ต่อเมื่อรันไฟล์นี้โดยตรง
    # (python main.py) แต่จะไม่ถูกเรียกอัตโนมัติถ้าไฟล์นี้ถูก import
    # ไปใช้ในไฟล์อื่น
    main()
