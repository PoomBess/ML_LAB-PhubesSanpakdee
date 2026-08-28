## Read data from CSV file and prepare for clustering mini-project

from pathlib import Path
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ชี้ไปยังไฟล์ characters.csv ที่มีอยู่ในโฟลเดอร์ data-Star Wars
CSV_PATH = Path(__file__).resolve().parent.parent / "data-Star Wars" / "characters.csv"

# Feature ที่ใช้จัดกลุ่ม (ดึงเฉพาะคอลัมน์ที่เป็นตัวเลข)
FEATURES = [
    "height",
    "mass"
]


# ----------------------------------------------------------------------
def load_data():
    """
    คืนค่าเป็น dict ที่มี
        X       : ข้อมูลหลัง scale แล้ว (ใช้จัดกลุ่ม)
        X_raw   : ข้อมูลหน่วยจริง (ใช้ตอนอธิบายผล)
        df      : ตารางเต็มจากไฟล์ CSV
    """
    df = pd.read_csv(CSV_PATH)
    
    # แปลงคอลัมน์ตัวเลขให้เป็น float และลบค่าว่าง (NaN)
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=FEATURES).reset_index(drop=True)

    X_raw = df[FEATURES].to_numpy(dtype="float32")
    X = StandardScaler().fit_transform(X_raw).astype("float32")

    return {"X": X, "X_raw": X_raw, "df": df, "features": FEATURES}


# ----------------------------------------------------------------------
if __name__ == "__main__":
    data = load_data()
    print("size data :", data["X"].shape)
    print("mean after scale (should be close to 0) :", data["X"].mean(axis=0).round(3))