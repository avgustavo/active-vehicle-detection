import argparse
import time
from ultralytics import YOLO

from pipeline import calculate_time
#################################### COnstantes #####################################
YAML_PATH = 'bdd100k/bdd100k.yaml'


def main(name: str):

    print('='*100)
    print(f'Treinamento completo {name}')
    print('='*100)

    model = YOLO('yolo11n.pt')
    t1 = time.time()
    model.train(
        data=YAML_PATH,
        epochs=15,
        imgsz=640,
        batch=32,
        device=[0, 1],
        project='runsbdd',
        name=name,
        plots=True,
        patience=5,
        optimizer='AdamW',
        lr0=0.0001,
        momentum=0.9,
        freeze=10
    )
    t2 = time.time()

    print(f'Tempo total de treinamento: {calculate_time(t1, t2)} segundos')

    best_model_p = f'runsbdd/{name}/weights/best.pt'
    model = YOLO(best_model_p)

    m_val = model.val(
        data=YAML_PATH,
        split='val',
        name=f'{name}_val',
        project='runsbdd',
    )
    print(f"  > mAP50-95 (val): {m_val.box.map:.4f}")
    print(f"  > mAP50 (val):    {m_val.box.map50:.4f}")
    print(f"  > mAP (val):      {m_val.box.map75:.4f}")

    results = model(source='bdd100k/images/test', batch=16, project='runsbdd', name='bdd100k_test', stream=True)

    i = 0
    for result in results:
        print(f"  > {i} - {result.probs}")
        result.save()
        i+=1
        if i > 10:
            break
    t3 = time.time()
    print(f'Tempo total de inferência: {calculate_time(t2, t3)} segundos')

    print('='*100)
    print('tempo total de execução: ', calculate_time(t1, t3), 'segundos')
    print('='*100)

if __name__ == '__main__':
    parse = argparse.ArgumentParser(description="Treinamento do modelo YOLO com o dataset BDD10K")
    parse.add_argument('--name', type=str, required=True, help="Nome do experimento")
    args = parse.parse_args()
    main(args.name)