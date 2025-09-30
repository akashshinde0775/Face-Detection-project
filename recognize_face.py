#recognize_face.py
from facenet_pytorch import MTCNN, InceptionResnetV1
import cv2, torch, pickle, os
import numpy as np
import argparse
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Argument parser
parser = argparse.ArgumentParser(description="Recognize face from webcam or image")
parser.add_argument('--image', type=str, help="Path to input image (optional)")
args = parser.parse_args()

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'
mtcnn = MTCNN(keep_all=False, device=device)  # Only detect one face
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Load embeddings
with open("embeddings.pkl", "rb") as f:
    database = pickle.load(f)

names, embeddings = [], []
for name, emb_list in database.items():
    for emb in emb_list:
        names.append(name)
        embeddings.append(emb)
embeddings = np.array(embeddings)

# Save full image with bounding box and label
def save_annotated_image(image, box, identity, confidence):
    os.makedirs("recognized_faces", exist_ok=True)
    x1, y1, x2, y2 = map(int, box)

    label = f"{identity} ({confidence:.1f}%)"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 100), 3)
    cv2.rectangle(image, (x1, y1 - 35), (x2, y1), (0, 255, 100), -1)
    cv2.putText(image, label, (x1 + 10, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"recognized_faces/{identity}_{timestamp}.jpg"
    cv2.imwrite(filename, image)
    print(f"✅ Saved full image with recognition: {filename}")

# Resize for display
def resize_frame(frame, max_width=800):
    h, w = frame.shape[:2]
    if w > max_width:
        scale = max_width / w
        return cv2.resize(frame, (int(w * scale), int(h * scale)))
    return frame

# ----------- 🔍 Image input mode --------------
if args.image:
    image = cv2.imread(args.image)
    if image is None:
        print("❌ Error: Could not load image.")
        exit()

    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    boxes, _ = mtcnn.detect(img_rgb)

    if face_tensor is not None and boxes is not None:
        with torch.no_grad():
            emb = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()
        sims = cosine_similarity(emb, embeddings)[0]
        max_idx = np.argmax(sims)
        confidence = float(sims[max_idx]) * 100
        identity = names[max_idx] if sims[max_idx] > 60 else "Unknown"

        save_annotated_image(image.copy(), boxes[0], identity, confidence)
        display = resize_frame(image)
        cv2.imshow("Face Recognition - Image", display)
        cv2.waitKey(0)
    else:
        print("❌ No face detected in the image.")
    cv2.destroyAllWindows()
    exit()

# ----------- 🎥 Webcam mode --------------
cap = cv2.VideoCapture(0)
print("🎥 Starting real-time face recognition. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_tensor = mtcnn(img_rgb)
    boxes, _ = mtcnn.detect(img_rgb)

    if face_tensor is not None and boxes is not None:
        with torch.no_grad():
            emb = resnet(face_tensor.unsqueeze(0).to(device)).cpu().numpy()
        sims = cosine_similarity(emb, embeddings)[0]
        max_idx = np.argmax(sims)
        confidence = float(sims[max_idx]) * 100
        identity = names[max_idx] if sims[max_idx] > 0.6 else "Unknown"

        annotated_frame = frame.copy()
        save_annotated_image(annotated_frame, boxes[0], identity, confidence)

        x1, y1, x2, y2 = map(int, boxes[0])
        label = f"{identity} ({confidence:.1f}%)"
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 100), 3)
        cv2.rectangle(frame, (x1, y1 - 35), (x2, y1), (0, 255, 100), -1)
        cv2.putText(frame, label, (x1 + 10, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)

    display = resize_frame(frame)
    cv2.imshow("Face Recognition - Webcam", display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
