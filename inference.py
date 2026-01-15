import cv2
import torch
import numpy as np
from torchvision import models, transforms
import mediapipe as mp

# ------------------ CONFIG ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 4
MODEL_PATH = "face_classifier.pt"
CAT_DIR = "cats"
CAM_ID = 0
# --------------------------------------------

# --- Load model ---
model = models.mobilenet_v2(weights=None)
model.classifier[1] = torch.nn.Linear(1280, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- Load cat images ---
cats = []
for i in range(NUM_CLASSES):
    img = cv2.imread(f"{CAT_DIR}/cat{i}.jpg")
    if img is None:
        raise FileNotFoundError(f"Missing {CAT_DIR}/cat{i}.jpg")
    cats.append(img)

# --- Face detector ---
mp_face = mp.solutions.face_detection.FaceDetection(
    model_selection=0,
    min_detection_confidence=0.6
)

# --- Webcam ---
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = mp_face.process(rgb)

    if results.detections:
        h, w, _ = frame.shape
        det = results.detections[0]
        box = det.location_data.relative_bounding_box

        x1 = max(0, int(box.xmin * w))
        y1 = max(0, int(box.ymin * h))
        x2 = min(w, int((box.xmin + box.width) * w))
        y2 = min(h, int((box.ymin + box.height) * h))

        face = frame[y1:y2, x1:x2]

        if face.size > 0:
            inp = transform(face).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(inp)
                pred = logits.argmax(dim=1).item()

            # Overlay cat
            cat = cv2.resize(cats[pred], (200, 200))
            frame[0:200, 0:200] = cat

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("face2cat", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
