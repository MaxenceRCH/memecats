import cv2

print("Scanning for available webcams...\n")

available_cameras = []

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✓ Camera {i} is available")
        available_cameras.append(i)
        cap.release()
    else:
        print(f"✗ Camera {i} is NOT available")

print(f"\nAvailable camera IDs: {available_cameras}")

if available_cameras:
    print(f"\nUse CAM_ID = {available_cameras[0]} in your inference.py")
else:
    print("\nNo cameras found!")
