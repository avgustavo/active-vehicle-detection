import argparse
from os import name
import time
from ultralytics import YOLO

from pipeline import calculate_time
#################################### COnstantes #####################################
YAML_PATH = 'treino_transitar/tt.yaml'

def printf(message: str):
    print('=' * 100)
    print(message)
    print('=' * 100)

def main(name: str):

    printf('Fine tuning do modelo yolo em treino_transitar')

    model_path = f'runstt/{name}/weights/best.pt'

    fname = f'{name}_fine_tune'

    printf(f'Fine tuning de {name}')

    model = YOLO(model_path)
    t1 = time.time()
    model.train(
        data=YAML_PATH,
        epochs=5,
        imgsz=640,
        batch=16,
        device=[0, 1],
        project='runstt',
        name=fname,
        plots=True,
        optimizer='AdamW',
        lr0=0.00001,
        momentum=0.9,
        # freeze=10,
        classes=[0, 1, 2, 3, 5, 6, 7, 9]
    )
    t2 = time.time()
    printf(f'Tempo total de treinamento: {calculate_time(t1, t2)} segundos')

    best_model_p = f'runstt/{fname}/weights/best.pt'
    model = YOLO(best_model_p)

    printf('Avaliação do modelo fine tuned')

    m_val = model.val(
        data=YAML_PATH,
        split='val',
        name=f'{fname}_val',
        project='runstt',
    )
    print(f"  > mAP50-95 (val): {m_val.box.map:.4f}")
    print(f"  > mAP50 (val):    {m_val.box.map50:.4f}")
    print(f"  > mAP (val):      {m_val.box.map75:.4f}")


    results = model(source='treino_transitar/val/images', batch=16, project='runstt', name=f'{fname}_pred', stream=True)


    i = 0
    for result in results:
        result.save()
        i+=1
        if i > 10:
            break
    t3 = time.time()
    printf(f'Tempo total de inferência: {calculate_time(t2, t3)} segundos')

    printf(f'Tempo total de execução: {calculate_time(t1, t3)} segundos')

if __name__ == '__main__':

    parse = argparse.ArgumentParser(description="Treinamento do modelo YOLO com o dataset treino_transitar")
    parse.add_argument('--name', type=str, required=True, help="Nome do experimento")
    args = parse.parse_args()
    main(args.name)