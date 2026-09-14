"""
evaluate.py
============
โมดูลสำหรับ "ประเมินผลโมเดล" (Step 6 ของ pipeline)

หน้าที่หลัก:
    - คำนวณ Accuracy โดยรวม
    - พิมพ์ Classification Report (precision / recall / f1-score ต่อคลาส)
    - คำนวณและวาด Confusion Matrix เป็นภาพ เพื่อดูว่าโมเดลสับสน/ทำนายผิด
      ระหว่างคลาสไหนกับคลาสไหนมากที่สุด

ทำไมดูแค่ Accuracy อย่างเดียวไม่พอ:
    Accuracy บอกแค่ "ถูกกี่ % โดยรวม" แต่ไม่บอกว่าความผิดพลาดกระจุกอยู่ที่
    คลาสไหน เช่น ถ้าทำนายคลาส A ผิดเป็น B บ่อยกว่าอีกทิศทางหนึ่งมาก ๆ
    Accuracy อย่างเดียวจะไม่เห็น pattern นี้ ต้องดู Confusion Matrix และ
    Precision/Recall แยกตามคลาสประกอบด้วยเสมอ
"""


import matplotlib

# ต้องเรียก matplotlib.use("Agg") ก่อน import pyplot เสมอ เพื่อบอกให้
# matplotlib วาดภาพแบบไม่ต้องพึ่งหน้าจอ (headless) เพราะเครื่องรันจริง
# (เช่น server/CI) อาจไม่มีหน้าจอแสดงผลให้ใช้
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


def evaluate_model(y_test, predictions, classes, save_path=None):
    """
    ประเมินผลโมเดลแบบครบชุด: accuracy, classification report, confusion matrix

    Args:
        y_test (array-like)      : label จริงของชุด test
        predictions (array-like) : label ที่โมเดลทำนายได้
        classes (list[str])      : ชื่อคลาสเรียงตาม index ของ label
        save_path (str | None)   : ถ้าระบุ path จะบันทึกภาพ confusion matrix
                                    ไว้ที่ path นี้

    Returns:
        float : ค่า accuracy โดยรวม (0-1)
    """

    # กำหนดลำดับ label ให้ตายตัว (0, 1, 2, ...) เพื่อให้ target_names
    # ใน classification_report ตรงกับคอลัมน์ของ confusion matrix เสมอ
    # ไม่ว่าข้อมูลจริงจะมีคลาสไหนปรากฏอยู่ในชุด test ครบหรือไม่ก็ตาม
    labels = list(range(len(classes)))

    # --- Accuracy: สัดส่วนที่ทำนายถูกจากทั้งหมด ---
    accuracy = accuracy_score(y_test, predictions)

    print("\n------------ Evaluation ------------------")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    # 👉 accuracy คือ % ที่ทายถูกโดยรวม แต่ยังไม่บอกว่าผิดพลาดกระจุกอยู่ที่คลาสไหน

    print("\nClassification Report:")

    # --- Classification Report: precision / recall / f1-score แยกตามคลาส ---
    #   precision = ในภาพที่โมเดลทายว่าเป็นคลาสนี้ ถูกจริงกี่ %
    #   recall    = ในภาพที่เป็นคลาสนี้จริง โมเดลทายถูกกี่ %
    #   zero_division=0 : ป้องกัน error/warning เวลาบางคลาสไม่มีตัวอย่าง
    #                      เลยในชุด test (ตั้งค่าคะแนนเป็น 0 แทนการ error)
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=classes,
        zero_division=0
    )

    print(report)
    print("Confusion Matrix:")
    # 👉 ดู precision/recall แยกรายคลาส เผื่อบางคลาสแม่นน้อยกว่าคลาสอื่นทั้งที่ accuracy รวมดูดี

    # --- Confusion Matrix: ตาราง (จริง x ทำนาย) นับจำนวนภาพในแต่ละคู่ ---
    # แนวตั้ง (แถว) = label จริง, แนวนอน (คอลัมน์) = label ที่ทำนายได้
    # เส้นทแยงมุมหลัก = จำนวนที่ทำนายถูก, ค่านอกแนวทแยง = ทำนายผิด/สับสน
    matrix = confusion_matrix(y_test, predictions, labels=labels)
    print(matrix)
    # 👉 ตัวเลขนอกแนวทแยงมุม = จำนวนภาพที่ทายผิด บอกได้ว่าโมเดลสับสนคลาสไหนกับคลาสไหนบ่อยสุด

    if save_path:
        plot_confusion_matrix(matrix, classes, save_path)
        print(f"Saved: {save_path}")

    return accuracy


def plot_confusion_matrix(matrix, classes, save_path):
    """
    วาด Confusion Matrix เป็นภาพ heatmap แล้วบันทึกเป็นไฟล์ .png

    Args:
        matrix (np.ndarray)  : confusion matrix ที่ได้จาก confusion_matrix()
        classes (list[str])  : ชื่อคลาสสำหรับกำกับแกน x และ y
        save_path (str)      : path ที่จะบันทึกไฟล์ภาพ
    """

    fig, ax = plt.subplots(figsize=(5, 5))
    # ใช้สี colormap "Blues": ค่ายิ่งมาก สียิ่งเข้ม ทำให้เห็น pattern
    # ของความผิดพลาด/ความแม่นยำได้ง่ายด้วยตาเปล่า
    ax.imshow(matrix, cmap="Blues")

    # กำกับชื่อคลาสบนแกน x (Predicted) และแกน y (True)
    ax.set_xticks(np.arange(len(classes)), classes)
    ax.set_yticks(np.arange(len(classes)), classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # เขียนตัวเลขจำนวนภาพลงในแต่ละช่องของตาราง โดยสลับสีตัวหนังสือ
    # (ขาว/ดำ) ตามความเข้มของพื้นหลัง เพื่อให้อ่านตัวเลขได้ชัดเจนทุกช่อง
    threshold = matrix.max() / 2
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, matrix[i, j], ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")
    # 👉 สลับสีตัวเลข ขาว/ดำ ตามความเข้มพื้นหลัง เพื่อให้อ่านตัวเลขได้ชัดทุกช่อง

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)  # ปิด figure เพื่อคืนหน่วยความจำ (สำคัญเวลาสร้างภาพหลายรูปในลูป)
