from ultralytics import YOLO
import cv2

# ====== Paths ======
model_path = "yolov8n.pt"  # Or your custom model
video_path = r"C:\Users\New Jawad Computers\Desktop\Internship\Virtual_line\Mall Enterance.mp4"
output_path = r"C:\Users\New Jawad Computers\Desktop\Internship\Virtual_line\Mall_Entrance_person_only.avi"

# ====== Load YOLO Model ======
model = YOLO(model_path)

# ====== Load Video ======
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("❌ Could not open video.")
    exit()

# ====== Video Properties ======
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ====== Process Frame by Frame ======
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(source=frame, imgsz=640, conf=0.4, verbose=False)[0]

    # Draw detections manually (only person class, which is class 0)
    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id == 0:  # Person
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Draw vertical center line
    center_x = width // 2
    cv2.line(frame, (center_x, 0), (center_x, height), (0, 255, 255), 2)

    # Show or save frame
    cv2.imshow("Only Person Detection", frame)
    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ====== Clean Up ======
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"✅ Output video saved at: {output_path}")
