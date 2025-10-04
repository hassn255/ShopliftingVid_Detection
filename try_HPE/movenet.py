import cv2
import numpy as np
import tensorflow as tf

# === CONFIG ===
MODEL_PATH = "4.tflite"  # your quantized model
INPUT_SIZE = 256  # matches your model (check shape in previous output)
CONFIDENCE_THRESHOLD = 0.2  # ignore weak detections

# === LOAD MODEL ===
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"✅ Loaded quantized MoveNet model | Input: {input_details[0]['shape']} | Type: {input_details[0]['dtype']}")

# === POSE CONNECTIONS (for skeleton lines) ===
POSE_CONNECTIONS = [
    (0, 1), (1, 3), (0, 2), (2, 4),
    (0, 5), (0, 6), (5, 7), (7, 9),
    (6, 8), (8, 10), (5, 11), (6, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

# === VIDEO CAPTURE ===
cap = cv2.VideoCapture("E:\courses and books\cellula_cv\week6\Shop DataSet\non shop lifters\shop_lifter_n_0.mp4")  # or replace with a path like "video.mp4"

if not cap.isOpened():
    print("❌ Error: Could not open video source.")
    exit()

print("🎥 Running MoveNet (quantized)... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    input_tensor = np.expand_dims(img_resized.astype(np.uint8), axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()

    # Get output
    keypoints = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # Draw keypoints and skeleton
    h, w, _ = frame.shape
    for i, kp in enumerate(keypoints):
        y, x, conf = kp
        if conf > CONFIDENCE_THRESHOLD:
            cx, cy = int(x * w), int(y * h)
            cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    for (p1, p2) in POSE_CONNECTIONS:
        if keypoints[p1][2] > CONFIDENCE_THRESHOLD and keypoints[p2][2] > CONFIDENCE_THRESHOLD:
            x1, y1 = int(keypoints[p1][1] * w), int(keypoints[p1][0] * h)
            x2, y2 = int(keypoints[p2][1] * w), int(keypoints[p2][0] * h)
            cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Show result
    cv2.imshow("MoveNet Pose Estimation", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()