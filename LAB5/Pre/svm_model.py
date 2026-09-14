"""
svm_model.py
=============
โมดูลหัวใจหลักของระบบ: สร้าง, ฝึก และใช้ทำนายด้วยโมเดล SVM (Step 4-5 ของ pipeline)

แนวคิดสำคัญ: "Scaler + PCA + SVM" ทำงานร่วมกันเป็น pipeline เดียว
    1) StandardScaler : ปรับทุก feature (พิกเซล) ให้มีค่าเฉลี่ย = 0 และ
       ส่วนเบี่ยงเบนมาตรฐาน = 1 เพื่อไม่ให้ feature ที่มีค่าตัวเลขสูงกว่า
       (เช่น พิกเซลบริเวณที่สว่าง) มีอิทธิพลต่อโมเดลมากเกินไปเมื่อเทียบกับ
       feature อื่น
    2) PCA (Principal Component Analysis) : ลดจำนวนมิติของข้อมูลจาก 10,000
       feature (ภาพ 100x100) ให้เหลือเพียง pca_components (ค่าเริ่มต้น 150)
       องค์ประกอบหลัก ช่วยให้:
           - SVM แบบ RBF kernel ฝึกได้เร็วขึ้นมาก (คำนวณน้อยลง)
           - ลด noise ที่ไม่เกี่ยวข้องกับการจำแนกภาพออกไป
       whiten=True ทำให้แต่ละองค์ประกอบหลังแปลงมีความแปรปรวนเท่ากัน (=1)
       ซึ่งช่วยให้ SVM ฝึกได้เสถียรขึ้น
    3) SVC (Support Vector Classifier) : ตัวโมเดลจำแนกจริง ใช้ RBF kernel
       ซึ่งเหมาะกับข้อมูลที่ขอบเขตระหว่างคลาสไม่ได้เป็นเส้นตรง (non-linear)

หมายเหตุสำคัญ: scaler (ในที่นี้คือ Pipeline ของ StandardScaler+PCA) ต้อง
fit กับข้อมูล train เท่านั้น แล้วนำ transform เดียวกันไปใช้กับข้อมูล test
(ห้าม fit ใหม่กับ test) เพื่อไม่ให้ข้อมูล test "รั่วไหล" (data leakage)
เข้าไปมีอิทธิพลต่อการเทรน
"""

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def train_svm(X_train, y_train, pca_components=150):
    """
    ฝึกโมเดล SVM โดยส่งข้อมูลผ่าน StandardScaler -> PCA -> SVC ตามลำดับ

    Args:
        X_train (np.ndarray)  : feature matrix ของชุด train
        y_train (np.ndarray)  : label ของชุด train
        pca_components (int)  : จำนวนองค์ประกอบหลักที่ PCA จะลดเหลือ

    Returns:
        model  : โมเดล SVC ที่ฝึกเสร็จแล้ว
        scaler : Pipeline ของ (StandardScaler + PCA) ที่ fit กับข้อมูล train
                 แล้ว ใช้ transform ข้อมูลใหม่ (เช่นตอนทำนาย) ให้อยู่ใน
                 รูปแบบเดียวกับตอนฝึก
    """
    # รวม StandardScaler และ PCA ไว้ใน Pipeline เดียวกัน เพื่อให้แน่ใจว่า
    # ข้อมูล test/ข้อมูลใหม่จะถูกแปลง (transform) ด้วยค่าพารามิเตอร์ชุดเดียว
    # กับที่ fit จากข้อมูล train เสมอ (ไม่มีทางลืม step ใด step หนึ่งไป)
    scaler = Pipeline([
        ("scaler", StandardScaler()),
        # min(pca_components, *X_train.shape) กันไม่ให้ตั้งค่า n_components
        # เกินจำนวน feature หรือจำนวนตัวอย่างที่มีจริง ซึ่งจะทำให้ PCA error
        ("pca", PCA(n_components=min(pca_components, *X_train.shape),
                    whiten=True, random_state=42)),
    ])
    # 👉 รวม 2 ขั้นตอนไว้ใน Pipeline เดียว ทำให้เรียก .transform() ครั้งเดียวได้ทั้งคู่ ไม่ต้องเขียน 2 บรรทัด

    # fit_transform: คำนวณค่าเฉลี่ย/PCA components จากข้อมูล train (fit)
    # แล้วแปลงข้อมูล train ชุดเดิมนั้นเลยในขั้นตอนเดียว (transform)
    X_train_scaled = scaler.fit_transform(X_train)
    # 👉 fit_transform = "เรียนรู้ค่าพารามิเตอร์จาก train" + "แปลงข้อมูล train เลยในทีเดียว"

    # สร้างโมเดล SVM แบบ RBF kernel
    #   kernel="rbf" : ใช้ Radial Basis Function เหมาะกับขอบเขตคลาสที่โค้ง/
    #                   ไม่เป็นเส้นตรง
    #   C=10         : ค่ายิ่งสูง โมเดลยิ่งพยายามจำแนกข้อมูล train ให้ถูก
    #                   มากที่สุด (เสี่ยง overfit ถ้าตั้งสูงเกินไป), ยิ่งต่ำ
    #                   โมเดลยิ่งยอมให้มีจุดที่จำแนกผิดได้มากขึ้นเพื่อแลกกับ
    #                   ขอบเขตการตัดสินใจที่ generalize ได้ดีกว่า
    #   gamma="scale": ให้ sklearn คำนวณค่า gamma อัตโนมัติจาก variance ของ
    #                   feature (ไม่ต้องตั้งเองแบบ manual)
    #   cache_size=1000 : ขนาด cache สำหรับ kernel matrix (หน่วย MB) ยิ่งมาก
    #                   ยิ่งฝึกเร็วขึ้นถ้ามี RAM เพียงพอ
    model = SVC(
        kernel="rbf", C=10, gamma="scale", cache_size=1000
    )

    # ฝึกโมเดลด้วยข้อมูลที่ผ่าน Scaler+PCA แล้ว
    model.fit(X_train_scaled, y_train)
    # 👉 ขั้นตอนนี้คือ "เรียนรู้" จริงๆ โมเดลหาขอบเขตที่แยกคลาสต่างๆ ออกจากกันให้ดีที่สุด

    return model, scaler


def predict_svm(model, scaler, X_test):
    """
    ใช้โมเดลที่ฝึกแล้วทำนาย label ของข้อมูลชุดใหม่ (เช่นชุด test)

    Args:
        model  : โมเดล SVC ที่ฝึกแล้วจาก train_svm()
        scaler : Pipeline (StandardScaler+PCA) ที่ fit ไว้แล้วจาก train_svm()
        X_test (np.ndarray) : feature matrix ของข้อมูลที่ต้องการทำนาย

    Returns:
        np.ndarray : label ที่โมเดลทำนายได้ สำหรับแต่ละตัวอย่างใน X_test
    """

    # ใช้ transform() เท่านั้น (ไม่ใช่ fit_transform) เพื่อนำค่าพารามิเตอร์
    # ที่ fit จากข้อมูล train มาแปลงข้อมูล test ให้อยู่ใน "สเกลเดียวกัน"
    # กับตอนฝึก ห้าม fit ใหม่กับข้อมูล test เด็ดขาด (data leakage)
    X_test_scaled = scaler.transform(X_test)
    # ทำนายผลลัพธ์ด้วยโมเดลที่ฝึกไว้แล้ว
    predictions = model.predict(X_test_scaled)

    return predictions
    # 👉 ใช้ .transform() เฉยๆ (ไม่ fit) เพราะต้องแปลงข้อมูล test ด้วยค่าพารามิเตอร์ชุดเดียวกับตอนฝึก train
