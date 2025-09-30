# register_face.py
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2, torch, pickle, os, time
import argparse
import numpy as np

# Argument for person's name
parser = argparse.ArgumentParser(description="Register a new face")
parser.add_argument('--name', required=True, help="Name of the person to register")
args = parser.parse_args()
name = args.name

# Load models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load or initialize database
if os.path.exists("embeddings.pkl"):
    with open("embeddings.pkl", "rb") as f:
        database = pickle.load(f)
else:
    database = {}

# Start camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Error: Webcam not accessible")
    exit()

required_samples = 10
collected = 0
last_capture_time = 0
capture_interval = 1  # seconds between samples

print(f"\n📸 Starting face registration for: {name}")
print("👉 Look directly at the camera. Slightly rotate your head for best results.\n")

try:
    while collected < required_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)

        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box)
                w, h = x2 - x1, y2 - y1

                # Only register reasonably sized faces
                if w < 80 or h < 80:
                    continue

                # Crop square box
                size = max(w, h)
                cx, cy = x1 + w // 2, y1 + h // 2
                half = size // 2
                x1_s, y1_s = max(0, cx - half), max(0, cy - half)
                x2_s, y2_s = cx + half, cy + half

                cropped_face = frame[y1_s:y2_s, x1_s:x2_s]
                img_rgb_cropped = cv2.cvtColor(cropped_face, cv2.COLOR_BGR2RGB)
                aligned_face = mtcnn(img_rgb_cropped)

                if aligned_face is not None:
                    current_time = time.time()
                    if current_time - last_capture_time > capture_interval:
                        with torch.no_grad():
                            embedding = resnet(aligned_face.unsqueeze(0).to(device)).cpu().numpy()[0]
                        database.setdefault(name, []).append(embedding)
                        collected += 1
                        last_capture_time = current_time
                        print(f"✅ Collected sample {collected}/{required_samples}")

                        # Draw square box to show collection
                        cv2.rectangle(frame, (x1_s, y1_s), (x2_s, y2_s), (0, 255, 0), 3)
                        cv2.putText(frame, f"Collected {collected}/{required_samples}",
                                    (x1_s, y1_s - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (0, 255, 0), 2)

        cv2.imshow("Registering Face", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("🛑 Interrupted by user.")
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()

    if collected > 0:
        with open("embeddings.pkl", "wb") as f:
            pickle.dump(database, f)
        print(f"\n✅ Successfully saved {collected} face embeddings for '{name}'.")
    else:
        print("\n❌ No face data collected.")
