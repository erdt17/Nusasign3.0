from ultralytics import YOLO

model_huruf = YOLO("best.pt")
print("Label model huruf:", model_huruf.names)

model_kata = YOLO("model_kata.pt")
print("Label model kata:", model_kata.names)
