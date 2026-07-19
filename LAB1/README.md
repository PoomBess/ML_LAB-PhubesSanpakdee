1. เลือก Dataset
ผมเลือกใช้ dataset ชื่อ "DC-MARVEL Comic Characters" จาก Kaggle ซึ่งเป็นข้อมูลตัวละครจากคอมมิค DC และ Marvel รวมกัน มีคอลัมน์สำคัญได้แก่:

Name ชื่อตัวละคร
Identity สถานะการปกปิดตัวตน
Alignment ฝ่าย (ดี/ร้าย/กลาง)
Eyes, Hair สีตา สีผม
Sex เพศ
Alive ยังมีชีวิตอยู่หรือไม่
Appearances จำนวนครั้งที่ปรากฏตัวในคอมิก
First_appeared วันที่ปรากฏตัวครั้งแรก
Universe สังกัด DC หรือ Marvel

เหตุผลที่เลือก dataset นี้ เพราะมีทั้งข้อมูลตัวเลข (Appearances) และข้อมูลหมวดหมู่ (Alignment, Sex, Universe) ผสมกัน เหมาะกับการฝึกทำ Data Preprocessing ครบทุกเทคนิคตามที่ใบงานกำหนด
2. เตรียมสภาพแวดล้อมสำหรับเขียนโค้ด
ผมใช้ VS Code ร่วมกับ Jupyter Notebook ในการเขียนโค้ด Python ระหว่างทางเจอปัญหาทางเทคนิค 2 เรื่องที่แก้ไปแล้ว:

ModuleNotFoundError — เครื่องยังไม่ได้ติดตั้ง library (pandas, numpy ฯลฯ) แก้ด้วยการ python -m pip install pandas numpy matplotlib seaborn scikit-learn
FileNotFoundError — โค้ดหาไฟล์ CSV ไม่เจอ เพราะไฟล์ไม่ได้อยู่ในโฟลเดอร์เดียวกับ notebook ที่กำลังรัน แก้โดยเช็ค working directory ด้วย os.getcwd() แล้วย้ายไฟล์ให้อยู่ที่เดียวกัน

3. วางแผนขั้นตอนการทำงานตามโครงสร้างใบงาน
ผมวางแผนแบ่งงานออกเป็น 4 ส่วนตามที่ใบงานกำหนด:
LAB1: Dataset Exploration — สำรวจข้อมูลก่อนแก้ไขใดๆ ได้แก่ โหลดข้อมูล ดูขนาด ดูชนิดข้อมูล ดูสถิติสรุป ตรวจค่าที่หายไป ตรวจข้อมูลซ้ำ และดูสัดส่วนของแต่ละ class
LAB2: Data Visualization — มองข้อมูลเป็นภาพด้วย Histogram (ดูการกระจายตัว) และ Correlation Heatmap (ดูความสัมพันธ์ระหว่างตัวแปร)
Part 3: Data Cleaning — ลงมือแก้ไขจริง ทั้งเติมค่าที่หายไปด้วย median/mode ลบข้อมูลซ้ำ แก้ข้อมูลที่ไม่สอดคล้องกัน (เช่นตัวพิมพ์ใหญ่เล็กปนกัน) แปลงชนิดข้อมูลให้ถูกต้อง และเปรียบเทียบค่า mean กับ median เพื่อดูว่าข้อมูลเบ้มากน้อยแค่ไหน
Part 4: Feature Engineering — แปลงข้อมูลข้อความให้เป็นตัวเลขด้วย Label Encoding (สำหรับข้อมูลที่มี 2 ค่าอย่าง Alive) และ One-Hot Encoding (สำหรับข้อมูลไม่มีลำดับอย่าง Alignment, Universe)
4. จัดโครงสร้าง GitHub Repository
ผมสร้าง repo ชื่อ ML_LAB-PhubesSanpakdee แล้วจัดระเบียบไฟล์โดยสร้างโฟลเดอร์ LAB1 เก็บไฟล์ที่เกี่ยวข้องกันไว้ในที่เดียว ได้แก่ ML_LAB.ipynb (โค้ด), README.md (คำอธิบายเนื้อหา), และ comic_characters.csv (ไฟล์ข้อมูล) เพื่อให้ repo เป็นระเบียบ แยกตามหัวข้องานชัดเจน
