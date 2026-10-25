# LAB8: VGG (Deep CNN) Image Recognition - Alien vs Predator

1. **Dataset Loading 
   * โหลดภาพชุดข้อมูลจากโฟลเดอร์โครงสร้างคลาสอัตโนมัติ (Alien และ Predator)
   * กำหนดขนาดภาพ (Image Size) และจำกัดจำนวนภาพต่อคลาสตามต้องการ

2. **Image Preprocessing 
   * ตรวจสอบความถูกต้องและแปลงรูปแบบช่องสีของภาพ (เช่น แปลงภาพ Grayscale หรือ BGR ให้เป็น RGB)
   * ปรับขนาดภาพ (Resize) ด้วยฟิลเตอร์ `cv2.INTER_AREA`
   * ทำ Normalization โดยปรับสเกลค่าพิกเซลให้อยู่ในช่วง $[0.0, 1.0]$ ด้วยการหาร `255.0` และแปลงข้อมูลเป็นชนิด `float32` เพื่อความเสถียรในการเทรนโมเดล

3. **Dataset Splitting 
   * แบ่งสัดส่วนข้อมูลออกเป็นชุดสำหรับฝึกฝน (Training Set), ชุดตรวจสอบความถูกต้อง (Validation Set) และชุดทดสอบ (Testing Set)

4. **Model Training & Architecture 
   * สร้างโมเดลสไตล์ VGG (รองรับทั้งโครงสร้างแบบเต็ม `VGG16` และแบบย่อ `VGG_SMALL` สำหรับใช้ทรัพยากรน้อย)
   * ดำเนินการฝึกโมเดล (Train) พร้อมบันทึกค่าน้ำหนักและประวัติการเรียนรู้ (History)

5. **Evaluation & Visualization 
   * ทดสอบประสิทธิภาพของโมเดลด้วยชุดข้อมูลทดสอบ (Test Set)
   * สร้างและบันทึกผลลัพธ์ประเมิน ได้แก่ **Confusion Matrix** และกราฟแสดงประวัติการฝึกซ้อม (Training History) ลงในโฟลเดอร์ `outputs/`

---


## Dataset
* **Dataset:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/pmigdal/alien-vs-predator-images)
