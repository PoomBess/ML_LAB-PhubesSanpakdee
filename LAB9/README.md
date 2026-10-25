# LAB9: R-CNN and Fast R-CNN Region Classification

โปรเจกต์นี้เป็นการศึกษาและพัฒนาโมเดล Deep Learning สำหรับการจำแนกประเภทวัตถุ (Object Classification) บนกรอบพิกัดวัตถุ (Bounding Boxes) โดยจำลองแนวคิดของสถาปัตยกรรม R-CNN และ Fast R-CNN พื้นฐาน

## โครงสร้างโปรเจกต์
- **`01-R-CNN/`**: โค้ดและชุดข้อมูลสำหรับการทดลองโมเดลจำแนกวัตถุแบบ R-CNN
- **`02-Fast-R-CNN/`**: โค้ดและชุดข้อมูลสำหรับการทดลองโมเดลจำแนกวัตถุแบบ Fast R-CNN ที่ใช้ ROI Pooling แบบประยุกต์

## ขั้นตอนการทำงานหลัก
1. **Train (`train.py`)**: 
   - โหลดภาพและตัดภาพวัตถุ (Crops) ตามพิกัดป้ายกำกับ YOLO
   - ปรับขนาดภาพให้เป็นมาตรฐาน ($64 \times 64$ พิกเซล)
   - เทรนโมเดล Convolutional Neural Network (CNN) ด้วย TensorFlow/Keras เพื่อจำแนกประเภทวัตถุ 10 คลาส
   - บันทึกน้ำหนักโมเดลไว้ที่ `outputs/model.keras`

2. **Evaluate (`evaluate.py`)**: 
   - โหลดโมเดลที่ฝึกสอนแล้วมาประเมินประสิทธิภาพกับชุดข้อมูล
   - คำนวณค่า Accuracy, Precision, Recall, F1-Score และแสดงผล Classification Report

3. **Inference / Test (`image_test.py`)**: 
   - ทดสอบการทำนายวัตถุบนภาพตัวอย่าง
   - วาดกรอบสี่เหลี่ยม (Bounding Box) พร้อมชื่อคลาสและค่าความมั่นใจ (Confidence Score) ลงบนภาพ และบันทึกผลลัพธ์ (`outputs/image_result.jpg`)

## คลาสวัตถุที่รองรับ (10 Classes)
`person`, `rider`, `car`, `bus`, `truck`, `bike`, `motor`, `traffic light`, `traffic sign`, `train`
## Dataset
* **Dataset:** [Kaggle Dataset Link](https://www.kaggle.com/datasets/ghulamhiader/yolo-ready-vehicle-images)
