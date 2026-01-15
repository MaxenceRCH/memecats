import cv2
import torch
import numpy as np
from torchvision import models, transforms

# ------------------ CONFIG ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 5
MODEL_PATH = "face_classifier.pt"
CAT_DIR = "cats"
CAM_ID = 1
# --------------------------------------------

# --- Load model ---
print("Loading model...")
model = models.resnet50(weights=None)
model.fc = torch.nn.Linear(2048, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval().to(DEVICE)

# --- Confidence threshold ---
CONFIDENCE_THRESHOLD = 0.4

# --- Cat name mapping ---
CAT_NAMES = {
    0: "Evil Larry",
    1: "Rigby",
    2: "Luna",
    3: "Uni",
    4: "Guangdang",
}

# --- Preprocessing ---
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225]),
])

# --- Load cat images ---
print("Loading cat images...")
cats = []
for i in range(NUM_CLASSES):
    img = cv2.imread(f"{CAT_DIR}/cat{i}.png")
    if img is None:
        raise FileNotFoundError(f"Missing {CAT_DIR}/cat{i}.jpg")
    cats.append(img)

# --- Webcam ---
print("Starting webcam...")
cap = cv2.VideoCapture(CAM_ID)
if not cap.isOpened():
    raise RuntimeError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    print("Processing frame...")    
    # Resize frame for model input
    resized_frame = cv2.resize(frame, (224, 224))
    inp = transform(resized_frame).unsqueeze(0).to(DEVICE)
    print("")
    with torch.no_grad():
        logits = model(inp)
        probs = torch.softmax(logits, dim=1)[0]  # Get probabilities for each class
        print(f"All class probabilities: {probs}")  # See if class 4 is always low
        confidence = torch.softmax(logits, dim=1).max().item()
        pred = logits.argmax(dim=1).item()
        print(f"Predicted class: {pred} with confidence {confidence:.4f}")

    # Only display if confidence is above threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        # Overlay cat
        cat = cv2.resize(cats[pred], (200, 200))
        frame[0:200, 0:200] = cat
        
        # Display confidence and prediction with cat name
        cat_name = CAT_NAMES[pred]
        cv2.putText(frame, f"Cat: {cat_name} | Conf: {confidence:.2f}", (10, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("face2cat", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
