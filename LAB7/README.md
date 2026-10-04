# LAB 7: Convolutional Neural Network (CNN)

## Description

ในแล็บนี้เป็นการศึกษาและสร้าง Convolutional Neural Network Pipeline ด้วยภาษา Python สำหรับงาน Image Recognition ซึ่งมีขั้นตอนการทำงานหลัก ดังนี้:

1. Image Loading & Preprocessing: นำเข้าไฟล์รูปภาพ และเตรียมข้อมูล เช่น การปรับขนาดรูปภาพ (Resizing), การแปลงระบบสี (BGR to RGB) และ Rescaling ค่าพิกเซล
2. Dataset Splitting: การแบ่งชุดข้อมูลออกเป็น Training Set, Validation Set และ Test Set เพื่อใช้ในการฝึกสอน ปรับสมดุล และวัดผล
3. Neural Network Training: การสร้างและฝึกสอนโมเดลโครงข่ายประสาทเทียมแบบ CNN (ประกอบด้วย Conv2D, Batch Normalization, Max Pooling, Global Average Pooling และ Data Augmentation) เพื่อเรียนรู้ฟีเจอร์ของรูปภาพ
4. Evaluation & Prediction: การประเมินประสิทธิภาพของโมเดลด้วย Accuracy, Classification Report และ Confusion Matrix รวมถึงนำโมเดลไปทดสอบทำนายผลรูปภาพใหม่
* **Dataset:** [Kaggle Dataset Link] https://www.kaggle.com/datasets/pmigdal/alien-vs-predator-images?resource=download
