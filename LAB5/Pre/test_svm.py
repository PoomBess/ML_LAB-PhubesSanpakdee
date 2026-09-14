"""
test_svm.py
============
สคริปต์สำหรับ "สุ่มตรวจสอบผลลัพธ์" ของโมเดลที่ฝึกไว้แล้ว (ไม่ได้อยู่ใน
ลำดับหลักของ main.py แต่เป็นเครื่องมือแยกไว้ใช้ตรวจสอบผลลัพธ์แบบรายภาพ
เพิ่มเติมจากตัวเลข accuracy/report ที่ evaluate.py แสดง)

หน้าที่:
    1) โหลดโมเดล, scaler และชุดข้อมูล test ที่ main.py บันทึกไว้แล้วใน
       โฟลเดอร์ outputs/ (ไม่ต้องรัน pipeline ทั้งหมดใหม่)
    2) สุ่มเลือกภาพจากชุด test มาจำนวน n_samples ภาพ (ไม่ fix seed จึงได้
       ภาพต่างชุดกันทุกครั้งที่รัน เพื่อช่วยตรวจสอบว่าโมเดล "เสถียร" จริง
       ไม่ใช่ดูดีแค่กับภาพบางชุดที่เคยตรวจสอบไปแล้ว)
    3) ทำนายผลของภาพที่สุ่มมา แล้ววาดเป็นภาพ grid พร้อมกำกับสีเขียว/แดง
       (ถูก/ผิด) และคำ Pred/True ใต้แต่ละภาพ เพื่อดูผลลัพธ์ด้วยตาได้ทันที
    4) บันทึกภาพ grid นี้เป็น outputs/prediction_sample.png
"""



import json

import joblib
import matplotlib

# ต้องตั้ง backend เป็น "Agg" ก่อน import pyplot เสมอ เพื่อให้วาดภาพได้
# แม้บนเครื่องที่ไม่มีหน้าจอ (headless environment)
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "outputs"
IMG_SIZE = 100     # ต้องตรงกับขนาดภาพตอนฝึก เพื่อ reshape กลับเป็นภาพ 2 มิติได้ถูกต้อง
N_SAMPLES = 4       # จำนวนภาพที่จะสุ่มมาทดสอบต่อการรัน 1 ครั้ง (ค่าเริ่มต้น)


def test_svm(n_samples=N_SAMPLES):
    """
    สุ่มภาพจากชุด test มาทำนายซ้ำ แล้ววาด/บันทึกผลเป็นภาพ grid

    Args:
        n_samples (int) : จำนวนภาพที่จะสุ่มมาทดสอบ
    """

    # ---- โหลดโมเดลและข้อมูลชุด test ที่ main.py บันทึกไว้แล้ว ----
    model = joblib.load(f"{OUTPUT_DIR}/svm_model.pkl")
    scaler = joblib.load(f"{OUTPUT_DIR}/scaler.pkl")
    X_test = np.load(f"{OUTPUT_DIR}/X_test.npy")
    y_test = np.load(f"{OUTPUT_DIR}/y_test.npy")
    with open(f"{OUTPUT_DIR}/classes.json") as f:
        classes = json.load(f)
    # 👉 โหลดทุกอย่างที่ main.py เคยเซฟไว้กลับมาใช้ ไม่ต้องรัน pipeline ทั้งหมดใหม่

    # ---- สุ่มเลือก index ภาพจากชุด test (ไม่ตั้ง seed -> สุ่มใหม่ทุกครั้ง) ----
    # replace=False กันไม่ให้สุ่มภาพซ้ำกันภายในชุดเดียวกัน
    index = np.random.choice(len(X_test), n_samples, replace=False)
    X_sample = X_test[index]
    y_sample = y_test[index]
    # 👉 ไม่ตั้ง seed ตรงนี้ ทุกครั้งที่รันจะได้ภาพชุดใหม่มาสุ่มตรวจ ช่วยดูว่าโมเดลเสถียรจริงไหม

    # ---- ทำนายผล ----
    # ต้อง transform ด้วย scaler ตัวเดียวกับตอนฝึกก่อนเสมอ (สเกลข้อมูล
    # ให้ตรงกับที่โมเดลเรียนรู้มา) ก่อนส่งเข้า model.predict()
    predictions = model.predict(scaler.transform(X_sample))
    # 👉 ต้อง transform ภาพที่สุ่มมาด้วย scaler ตัวเดิมก่อน ให้สเกลตรงกับตอนโมเดลเรียนรู้มา

    # ---- จัดวางภาพผลลัพธ์เป็นตาราง (grid) ขนาดประมาณสี่เหลี่ยมจัตุรัส ----
    # เช่น n_samples=4 -> cols=2, rows=2 (grid 2x2)
    cols = int(np.ceil(np.sqrt(n_samples)))
    rows = int(np.ceil(n_samples / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 4.0 * rows))
    # atleast_1d + ravel: ทำให้ axes เป็น array 1 มิติเสมอ ไม่ว่า rows/cols
    # จะเป็น 1 หรือมากกว่า (ปกติ plt.subplots คืน axes เป็น object เดี่ยว
    # ถ้ามีแค่ช่องเดียว ทำให้ index/loop ต่อไปนี้พังถ้าไม่แปลงก่อน)
    axes = np.atleast_1d(axes).ravel()

    for i, ax in enumerate(axes):
        # กรณี grid มีช่องเหลือมากกว่าจำนวนภาพจริง (เช่น 3 ภาพใน grid 2x2)
        # ให้ปิดแกนของช่องที่ไม่ได้ใช้ไปเฉย ๆ
        if i >= n_samples:
            ax.axis("off")
            continue

        pred = classes[predictions[i]]   # แปลง label ตัวเลขที่ทำนายได้ -> ชื่อคลาส
        true = classes[y_sample[i]]      # แปลง label ตัวเลขจริง -> ชื่อคลาส
        correct = predictions[i] == y_sample[i]
        color = "green" if correct else "red"   # เขียว=ถูก, แดง=ผิด

        # reshape เวกเตอร์ภาพ (IMG_SIZE*IMG_SIZE,) กลับเป็นภาพ 2 มิติ
        # (IMG_SIZE, IMG_SIZE) เพื่อแสดงผลด้วย imshow
        ax.imshow(X_sample[i].reshape(IMG_SIZE, IMG_SIZE), cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Pred: {pred}\nTrue: {true}", color=color)

        print(f"[{i + 1}] Pred: {pred:<6} True: {true:<6} "
              f"{'OK' if correct else 'WRONG'}")
    # 👉 สีเขียว = โมเดลทายถูก, สีแดง = ทายผิด ดูผลแบบภาพจริงได้เร็วกว่าดูแค่ตัวเลข

    # นับจำนวนที่ทำนายถูกทั้งหมดในรอบนี้ แสดงเป็นหัวข้อรวมของภาพ grid
    correct_total = int((predictions == y_sample).sum())
    print(f"\nCorrect: {correct_total}/{n_samples}")

    fig.suptitle(f"Prediction: {correct_total}/{n_samples} correct")
    fig.tight_layout()

    # บันทึกภาพ grid ผลลัพธ์ลงไฟล์
    save_path = f"{OUTPUT_DIR}/prediction_sample.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)  # ปิด figure เพื่อคืนหน่วยความจำ
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    # รันฟังก์ชันทดสอบเมื่อเรียกไฟล์นี้โดยตรง (python test_svm.py)
    test_svm()
