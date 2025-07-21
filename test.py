import os
from pathlib import Path
from ultralytics import YOLO

def main():

    model = YOLO('yolo11n.pt')

    files = []

    abs_path = Path('FOCAL/yolov5_format/images/train').absolute()

    with open('/home/avgus/active-vehicle-detection/FOCAL/image_paths.txt', 'r') as f:

        i = 0 
        for line in f:
            files.append(os.path.join(str(abs_path), line.strip()))
            i += 1
            if i == 32:
                break
    
    results = model(files, stream=True, batch=8, verbose=True, device=[0, 1])

    for result in results:
        print(f"Image: {result.path}")
        print(f"Predictions: {result.pred[0].boxes.xyxy}")
        print(f"Classes: {result.pred[0].boxes.cls}")
        print(f"Scores: {result.pred[0].boxes.conf}")


if __name__ == "__main__":
    main()
    # main()

    
