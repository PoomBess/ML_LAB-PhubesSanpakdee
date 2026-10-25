from pathlib import Path
import argparse
import numpy as np
import cv2
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import json

# Paths (แก้ไขให้ ROOT และ DATASET ชี้มาที่โฟลเดอร์ภายใน 02-Fast-R-CNN โดยตรง)
ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "dataset"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_PATH = OUT / "model.keras"

NAMES = ["person", "rider", "car", "bus", "truck", "bike", "motor", "traffic light", "traffic sign", "train"]
IMG_SIZE = 64

def preprocess(crop):
    return cv2.cvtColor(cv2.resize(crop, (IMG_SIZE, IMG_SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32)/255.0

def load_crops(split, limit):
    images_dir = DATASET / split / 'images'
    labels_dir = DATASET / split / 'labels'
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f'Missing dataset folders: {images_dir} or {labels_dir}')
    files = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    if limit > 0: files = files[:limit]
    xs, ys = [], []
    for path in files:
        label = labels_dir / (path.stem + '.txt')
        if not label.exists(): continue
        im = cv2.imread(str(path))
        if im is None: continue
        h,w = im.shape[:2]
        for line in label.read_text(encoding='utf-8').splitlines():
            parts=line.split()
            if len(parts)!=5: continue
            try:
                cls=int(parts[0]); xc,yc,bw,bh=map(float,parts[1:])
            except ValueError: continue
            if not 0 <= cls < len(NAMES): continue
            x1=max(0,int((xc-bw/2)*w)); y1=max(0,int((yc-bh/2)*h))
            x2=min(w,int((xc+bw/2)*w)); y2=min(h,int((yc+bh/2)*h))
            if x2<=x1 or y2<=y1: continue
            xs.append(preprocess(im[y1:y2,x1:x2])); ys.append(cls)
    if not xs: raise RuntimeError(f'No labeled crops in {images_dir}. Check image names and YOLO labels.')
    return np.asarray(xs,np.float32),np.asarray(ys,np.int32)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--limit', type=int, default=200)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='train')  # เปลี่ยนค่าเริ่มต้นเป็น train
    p.add_argument('--batch', type=int, default=32)
    args = p.parse_args()

    x, y = load_crops(args.split, args.limit)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f'Model not found: {MODEL_PATH}')

    model = tf.keras.models.load_model(MODEL_PATH)
    pred = model.predict(x, batch_size=args.batch, verbose=1).argmax(axis=1)

    precision, recall, f1, _ = precision_recall_fscore_support(y, pred, average='macro', labels=list(range(len(NAMES))), zero_division=0)
    metrics = {
        'split': args.split,
        'crops': len(y),
        'accuracy': float(accuracy_score(y, pred)),
        'precision_macro': float(precision),
        'recall_macro': float(recall),
        'f1_macro': float(f1)
    }

    print(json.dumps(metrics, indent=2))
    print(classification_report(y, pred, labels=list(range(len(NAMES))), target_names=NAMES, zero_division=0))
    
    (OUT / 'evaluation.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    print('Crop classification only: detection mAP is NOT calculated.')

if __name__ == '__main__':
    main()