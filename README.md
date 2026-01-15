# MemeCats - Cat Classification with Live Webcam Feed

A real-time cat classification system using ResNet-50 that processes webcam feeds and overlays cat images based on predictions. Perfect to know which instagram cat you look the most like !

## 🎯 Features

- **Live Webcam Processing**: Real-time cat classification from your webcam
- **Cat Image Overlay**: Shows a custom cat image when a prediction is made
- **Transfer Learning**: Uses pretrained ResNet-50

## 📋 Project Structure

```
memecats/
├── detect_cameras.py      # Scan and preview available cameras
├── train.py               # Train the cat classification model
├── inference.py           # Run live inference on webcam feed
├── requirements.txt       # Python dependencies
├── face_classifier.pt     # Trained model weights (generated after training)
├── cats/                  # Directory containing cat images (cat0.png - cat4.png)
├── data/                  # Training data directory (organized by class)
└── venv/                  # Virtual environment (created during setup)
```

## 🚀 Setup

### 1. Create and Activate Virtual Environment

```powershell
cd c:\AMOA\memecats
python -m venv venv
venv\Scripts\Activate.ps1
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

Dependencies include:
- torch
- torchvision
- opencv-python
- numpy
- tqdm

## 🎮 Usage

### 1. Detect Available Cameras

Before running inference, check which cameras are available on your system:

```powershell
python detect_cameras.py
```

This will Scan for available cameras (0-9), helping to choose the correct camera for inference.

### 2. Prepare Training Data

Organize your training data in the `data/` folder by class:

```
data/
├── cat0/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
├── cat1/
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ...
└── cat2/
    └── ...
```

### 3. Train the Model

```powershell
python train.py
```

Configuration options in `train.py`:
- `BATCH_SIZE`: 32 (training batch size)
- `EPOCHS`: 8 (number of training epochs)
- `LR`: 1e-3 (learning rate)
- `VAL_SPLIT`: 0.2 (validation split ratio)
- `NUM_CLASSES`: 5 (number of cat classes)

The trained model will be saved as `face_classifier.pt`

### 4. Prepare Cat Images

Place cat images in the `cats/` folder named as `cat0.png`, `cat1.png`, etc. (matching your NUM_CLASSES). These images will be overlaid on the webcam feed when detected.

### 5. Run Live Inference

```powershell
python inference.py
```

Configuration options in `inference.py`:
- `CAM_ID`: Which camera to use (default: 1)
- `NUM_CLASSES`: Number of cat classes (default: 5)
- `CONFIDENCE_THRESHOLD`: Minimum confidence to display prediction (default: 0.6)
- `CAT_NAMES`: Dictionary mapping class IDs to friendly names

Controls:
- Press **ESC** to exit

## ⚙️ Configuration

### Camera Selection
Edit `CAM_ID` in `inference.py` to use a different camera (run `detect_cameras.py` to find the right ID)

### Confidence Threshold
Adjust `CONFIDENCE_THRESHOLD` in `inference.py` to filter out low-confidence predictions (0.0 - 1.0)

### Cat Names
Update the `CAT_NAMES` dictionary in `inference.py` to display custom names:

```python
CAT_NAMES = {
    0: "Evil Larry",
    1: "Rigby",
    2: "Luna",
    3: "Uni",
    4: "Guangdang",
}
```

### Model Architecture
To change the model, edit the model loading section in both `train.py` and `inference.py`. Available options:
- ResNet-50 (current, recommended)
- ResNet-101
- EfficientNet
- ViT (Vision Transformer)

## 🔍 Troubleshooting

**Low accuracy:**
- Collect more training data
- Increase `EPOCHS` in `train.py`
- Adjust `CONFIDENCE_THRESHOLD` to filter uncertain predictions

## 📊 Model Details

- **Architecture**: ResNet-50
- **Input Size**: 224x224 pixels
- **Pre-trained Weights**: ImageNet (for transfer learning)
- **Optimizer**: Adam
- **Loss Function**: Cross-Entropy Loss

## 📦 Requirements

- Python 3.8+
- CUDA 11.8+ (optional, for GPU acceleration)
- 8GB+ RAM recommended

## 📝 License

This project is open source and available for personal and educational use.
