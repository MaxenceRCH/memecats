import cv2
import time

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
    print("\n" + "="*50)
    print("Starting camera preview...")
    print("Press 'q' to exit, 'n' for next camera")
    print("="*50 + "\n")
    
    for cam_id in available_cameras:
        print(f"\nOpening Camera {cam_id}...")
        cap = cv2.VideoCapture(cam_id)
        
        if not cap.isOpened():
            print(f"Failed to open camera {cam_id}")
            continue
        
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        window_name = f"Camera {cam_id} - Press 'n' for next or 'q' to quit"
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print(f"Failed to read from camera {cam_id}")
                break
            
            # Add camera ID and FPS info to the frame
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            cv2.putText(frame, f"Camera ID: {cam_id}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            cv2.imshow(window_name, frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('n'):
                print(f"Switching to next camera...")
                break
            elif key == ord('q'):
                print("Exiting camera preview.")
                cap.release()
                cv2.destroyAllWindows()
                exit()
        
        cap.release()
        cv2.destroyAllWindows()
else:
    print("\nNo cameras found!")
