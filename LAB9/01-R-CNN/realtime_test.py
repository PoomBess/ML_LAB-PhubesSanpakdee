import argparse
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
MODEL_PATH = OUT / "model.keras"

CLASSES = [
    "person", "rider", "car", "bus", "truck",
    "bike", "motor", "traffic light", "traffic sign", "train"
]

IMG_SIZE = 64


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    return tf.keras.models.load_model(MODEL_PATH)


def detect_demo(model, frame, threshold=0.85, regions=25):
    h, w = frame.shape[:2]
    crops = []
    boxes = []

    # สร้างกริด Sliding Window เบื้องต้นสำหรับจำแนกภาพจากกล้อง
    grid_size = int(np.sqrt(regions))
    step_h = h // (grid_size + 1)
    step_w = w // (grid_size + 1)

    for i in range(1, grid_size + 1):
        for j in range(1, grid_size + 1):
            cx = j * step_w
            cy = i * step_h
            box_w = w // (grid_size + 1)
            box_h = h // (grid_size + 1)

            x1 = max(0, cx - box_w // 2)
            y1 = max(0, cy - box_h // 2)
            x2 = min(w, cx + box_w // 2)
            y2 = min(h, cy + box_h // 2)

            if x2 - x1 > 10 and y2 - y1 > 10:
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crops.append(crop)
                boxes.append((x1, y1, x2, y2))

    if not crops:
        return []

    x = np.asarray(crops, dtype=np.float32) / 255.0
    predictions = model.predict(x, verbose=0)

    results = []
    for box, pred in zip(boxes, predictions):
        cls = int(np.argmax(pred))
        confidence = float(np.max(pred))
        if confidence >= threshold:
            results.append((box, cls, confidence))

    return results


def draw(frame, preds):
    for box, cls, confidence in preds:
        x1, y1, x2, y2 = box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{CLASSES[cls]} {confidence:.2f}",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, (0, 255, 0), 2
        )
    return frame


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--camera', type=int, default=0)
    p.add_argument('--threshold', type=float, default=0.85)
    p.add_argument('--regions', type=int, default=25, help='Smaller is faster on CPU')
    args = p.parse_args()

    model = load_model()
    cap = cv2.VideoCapture(args.camera)

    if not cap.isOpened():
        raise RuntimeError(f'Cannot open camera {args.camera}')

    print('Press Q to quit. Sliding-window crop classification demo, not real-time RPN detection.')

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            preds = detect_demo(model, frame, args.threshold, args.regions)
            cv2.imshow('Region classifier demo (Q to quit)', draw(frame, preds))

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()