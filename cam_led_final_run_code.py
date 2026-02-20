import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
import RPi.GPIO as GPIO

MODEL_PATH = "/home/prashant/road_model.tflite"
THRESHOLD = 0.52
LED_PIN = 17

# ---------------- GPIO Setup ----------------
GPIO.setmode(GPIO.BCM)
GPIO.setup(LED_PIN, GPIO.OUT)
GPIO.output(LED_PIN, GPIO.LOW)

# ---------------- Load Model ----------------
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- Start Camera ----------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not detected")
    GPIO.cleanup()
    exit()

print("Live detection started... Press 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # -------- Preprocess --------
    img = cv2.resize(frame, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=0).astype(np.float32)

    # -------- Inference --------
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    # -------- Classification --------
    if prediction > THRESHOLD:
        label = "NORMAL"
        color = (0, 255, 0)
        GPIO.output(LED_PIN, GPIO.LOW)
    else:
        label = "ANOMALY"
        color = (0, 0, 255)
        GPIO.output(LED_PIN, GPIO.HIGH)

    # -------- Display --------
    cv2.putText(frame, f"{label} ({prediction:.2f})",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2)

    cv2.imshow("Road Anomaly Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
GPIO.cleanup()
cv2.destroyAllWindows()