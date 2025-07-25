import argparse
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

    printf('Avaliação do modelo yolo em treino_transitar')

    model = YOLO('yolo11n.pt')

    results_0 = model.val(
        data=YAML_PATH,
        split='val',
        name=f'{name}_11n_test',
        project='runstt',
        classes=[0, 1, 2, 3, 5, 6, 7],
    )

    res_csv = results_0.to_csv()
    with open(f'runstt/{name}_11n_test/results.csv', 'w') as f:
        f.write(res_csv)

    printf(f'Treinamento completo em treino_transitar {name}')
    model = YOLO('yolo11n.pt')
    t1 = time.time()
    model.train(
        data=YAML_PATH,
        epochs=25,
        imgsz=640,
        batch=16,
        device=[0, 1],
        project='runstt',
        name=name,
        plots=True,
        # optimizer='AdamW',
        # lr0=0.0001,
        # momentum=0.9,
        # freeze=10,
        classes=[0, 1, 2, 3, 5, 6, 7]
    )
    t2 = time.time()
    printf(f'Tempo total de treinamento: {calculate_time(t1, t2)} segundos')

    best_model_p = f'runstt/{name}/weights/best.pt'
    model = YOLO(best_model_p)

    m_val = model.val(
        data=YAML_PATH,
        split='val',
        name=f'{name}_val',
        project='runstt',
    )
    print(f"  > mAP50-95 (val): {m_val.box.map:.4f}")
    print(f"  > mAP50 (val):    {m_val.box.map50:.4f}")
    print(f"  > mAP (val):      {m_val.box.map75:.4f}")

    m_val_csv = m_val.to_csv()
    with open(f'runstt/{name}/res_val.csv', 'w') as f:
        f.write(m_val_csv)

    t3 = time.time()
    printf(f'Tempo total de validação: {calculate_time(t2, t3)} segundos')
    printf(f'Treinamento e validação completos em treino_transitar {name} em {calculate_time(t1, t3)}')


if __name__ == '__main__':

    parse = argparse.ArgumentParser(description="Treinamento do modelo YOLO com o dataset BDD10K")
    parse.add_argument('--name', type=str, required=True, help="Nome do experimento")
    args = parse.parse_args()
    main(args.name)