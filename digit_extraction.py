import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import cv2
import numpy as np
import tensorflow as tf

def preprocess_digit(crop):
    """Preprocess a cropped digit image matching the training pipeline."""
    if len(crop.shape) == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    crop = cv2.resize(crop, (64, 64))
    crop = cv2.medianBlur(crop, 3)
    crop = cv2.adaptiveThreshold(
        crop, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    kernel = np.ones((2, 2), np.uint8)
    crop = cv2.morphologyEx(crop, cv2.MORPH_OPEN, kernel)
    return crop.astype(np.float32) / 255.0

def solve_captcha(image_path, model_path=None):
    if not os.path.exists(image_path):
        print(f"[ERROR] Captcha image not found: {image_path}")
        return None

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter bounding boxes with a minimum size to avoid noise
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w >= 15 and h >= 25:
            boxes.append((x, y, w, h))

    # Sort boxes from left to right
    boxes = sorted(boxes, key=lambda b: b[0])
    print(f"[INFO] Found {len(boxes)} digit segments in captcha.")

    # Load model if available
    model = None
    if model_path and os.path.exists(model_path):
        try:
            model = tf.keras.models.load_model(model_path)
            print(f"[INFO] Loaded digit recognition model from: {model_path}")
        except Exception as e:
            print(f"[WARNING] Could not load model: {e}")

    recognized_digits = []
    annotated = img.copy()

    for i, (x, y, w, h) in enumerate(boxes):
        pad = 5
        y1 = max(0, y - pad)
        y2 = min(img.shape[0], y + h + pad)
        x1 = max(0, x - pad)
        x2 = min(img.shape[1], x + w + pad)

        crop = gray[y1:y2, x1:x2]
        pred_digit = "?"

        if model is not None:
            prepped = preprocess_digit(crop).reshape(1, 64, 64, 1)
            probs = model.predict(prepped, verbose=0)
            pred_idx = int(np.argmax(probs))
            # Model output classes are digits 1..9 (indices 0..8)
            pred_digit = str(pred_idx + 1)
            recognized_digits.append(pred_digit)

        # Draw bounding rectangle
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put predicted label above rectangle
        cv2.putText(
            annotated,
            str(pred_digit),
            (x1 + 5, y1 - 8 if y1 - 8 > 10 else y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    code = "".join(recognized_digits) if recognized_digits else "N/A"
    print(f"[RESULT] Solved Captcha Text: {code}")

    # Save output visualization
    out_path = os.path.join(os.path.dirname(image_path), "captcha_result.png")
    cv2.imwrite(out_path, annotated)
    print(f"[INFO] Annotated visualization saved to: {out_path}")

    return code, annotated

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    img_path = os.path.join(base_dir, "CAptcha.jpg")
    model_path = os.path.join(base_dir, "digit_best.keras")

    if not os.path.exists(model_path):
        # Fallback to iris dataset folder if present
        fallback_model = os.path.abspath(os.path.join(base_dir, "..", "iris dataset", "digit_best.keras"))
        if os.path.exists(fallback_model):
            model_path = fallback_model

    code, annotated = solve_captcha(img_path, model_path)