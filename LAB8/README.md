# LAB8: VGG (Deep CNN) Image Recognition - Alien vs Predator

1. **Dataset Loading (`data_loader.py`)**[cite: 2]
   * โหลดภาพชุดข้อมูลจากโฟลเดอร์โครงสร้างคลาสอัตโนมัติ (Alien และ Predator)[cite: 5]
   * กำหนดขนาดภาพ (Image Size) และจำกัดจำนวนภาพต่อคลาสตามต้องการ

2. **Image Preprocessing (`preprocessing.py`)**[cite: 6]
   * ตรวจสอบความถูกต้องและแปลงรูปแบบช่องสีของภาพ (เช่น แปลงภาพ Grayscale หรือ BGR ให้เป็น RGB)[cite: 6]
   * ปรับขนาดภาพ (Resize) ด้วยฟิลเตอร์ `cv2.INTER_AREA`[cite: 6]
   * ทำ Normalization โดยปรับสเกลค่าพิกเซลให้อยู่ในช่วง $[0.0, 1.0]$ ด้วยการหาร `255.0` และแปลงข้อมูลเป็นชนิด `float32` เพื่อความเสถียรในการเทรนโมเดล

3. **Dataset Splitting (`split_data.py`)**[cite: 3]
   * แบ่งสัดส่วนข้อมูลออกเป็นชุดสำหรับฝึกฝน (Training Set), ชุดตรวจสอบความถูกต้อง (Validation Set) และชุดทดสอบ (Testing Set)

4. **Model Training & Architecture (`vgg_model.py`)**[cite: 3]
   * สร้างโมเดลสไตล์ VGG (รองรับทั้งโครงสร้างแบบเต็ม `VGG16` และแบบย่อ `VGG_SMALL` สำหรับใช้ทรัพยากรน้อย)[cite: 3]
   * ดำเนินการฝึกโมเดล (Train) พร้อมบันทึกค่าน้ำหนักและประวัติการเรียนรู้ (History)

5. **Evaluation & Visualization (`evaluate.py`)**[cite: 3]
   * ทดสอบประสิทธิภาพของโมเดลด้วยชุดข้อมูลทดสอบ (Test Set)[cite: 3]
   * สร้างและบันทึกผลลัพธ์ประเมิน ได้แก่ **Confusion Matrix** และกราฟแสดงประวัติการฝึกซ้อม (Training History) ลงในโฟลเดอร์ `outputs/`[cite: 3, 6]

---

## 🚀 วิธีการรันโปรเจกต์
1. เตรียมชุดข้อมูลภาพไว้ในโฟลเดอร์ `dataset/` โดยภายในต้องมีโฟลเดอร์ย่อยชื่อ `alien` และ `predator`[cite: 5]
2. ติดตั้งไลบรารีที่จำเป็น (เช่น `opencv-python`, `numpy`, `scikit-learn`, `tensorflow/keras`, `matplotlib`)
3. รันไฟล์หลักของโปรเจกต์เพื่อเริ่มกระบวนการตั้งแต่โหลดข้อมูล, เทรน จนถึงประเมินผล:
   ```bash
   python main.py

## Dataset
* **Dataset:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/pmigdal/alien-vs-predator-images)
