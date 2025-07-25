import argparse
import time
from ultralytics import YOLO

from pipeline import calculate_time
#################################### COnstantes #####################################
YAML_PATH = 'd10k/bdd100k.yaml'


def main(name: str):

    print('='*100)
    print(f'Fine Tuning {name}')
    print('='*100)

    model_path = 'bddruns/b10k_freeze/weights/best.pt'

    model = YOLO(model_path)
    t1 = time.time()
    model.train(
        data=YAML_PATH,
        epochs=10,
        imgsz=640,
        batch=16,
        device=[0, 1],
        project='runsbdd',
        name=name,
        plots=True,
        optimizer='AdamW',
        lr0=0.0001,
        momentum=0.9,
        # freeze=10
    )
    t2 = time.time()
    print(f'Tempo total de treinamento: {calculate_time(t1, t2)} segundos')

if __name__ == '__main__':

    parse = argparse.ArgumentParser(description="Treinamento do modelo YOLO com o dataset BDD10K")
    parse.add_argument('--name', type=str, required=True, help="Nome do experimento")
    args = parse.parse_args()
    main(args.name)