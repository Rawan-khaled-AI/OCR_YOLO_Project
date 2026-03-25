from ultralytics import YOLO
from pathlib import Path
import cv2

# مسار الموديل
model_path = Path(r"E:\project_ocr\medicine.v3i.yolov8\runs\detect\train\weights\best.pt")

# حمّل الموديل
model = YOLO(str(model_path))

# صورة اختبار
image_path = r"E:\project_ocr\medicine.v3i.yolov8\train\images\0a04b773a9854e188133014b73b30366_jpg.rf.c576bdb7719c6ff20528680ddf235436.jpg"

# شغل detection
results = model(image_path, conf=0.25)

# احفظ الصورة بالبوكسات
results[0].save(filename="yolo_test_result.jpg")

print("Detection done ✅ Check yolo_test_result.jpg")