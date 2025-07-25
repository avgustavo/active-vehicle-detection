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

    printf('Avaliação do modelo treino_transitar no FOCAL')

    best_model_p = f'runstt/tt02/weights/best.pt'
    model = YOLO(best_model_p)

    m_val = model.val(
        data=YAML_PATH,
        split='val',
        name=f'{name}_val',
        project='runstt',
        classes=[0, 1, 2],
    )
    print(f"  > mAP50-95 (val): {m_val.box.map:.4f}")
    print(f"  > mAP50 (val):    {m_val.box.map50:.4f}")
    print(f"  > mAP (val):      {m_val.box.map75:.4f}")

    m_val_csv = m_val.to_csv()
    with open(f'runstt/{name}_val/res_val.csv', 'w') as f:
        f.write(m_val_csv)




if __name__ == '__main__':

    parse = argparse.ArgumentParser(description="Treinamento do modelo YOLO com o dataset BDD10K")
    parse.add_argument('--name', type=str, required=True, help="Nome do experimento")
    args = parse.parse_args()
    main(args.name)