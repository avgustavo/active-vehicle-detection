import time
from ultralytics import YOLO

from pipeline import calculate_time
#################################### COnstantes #####################################
YAML_PATH = 'd10k/bdd10k.yaml'


def main():

    # print('Treinamento completo bdd10k')

    model = YOLO('yolo11n.pt')
    t1 = time.time()
    model.train(
        data=YAML_PATH,
        epochs=100,
        imgsz=640,
        batch=16,
        device=[0, 1],
        project='runsbdd',
        name='bdd10k_100_epochs',
        plots=True,
        patience=15,
    )
    t2 = time.time()
    print(f'Tempo total de treinamento: {calculate_time(t1, t2)} segundos')

    best_model_p = 'runsbdd/bdd10k/weights/best.pt'
    model = YOLO(best_model_p)

    m_val = model.val(
        data=YAML_PATH,
        split='val',
        name='bdd10k_100_epochs_val',
        project='runsbdd',
        save_json=True,
    )
    print(f"  > mAP50-95 (val): {m_val.box.map:.4f}")
    print(f"  > mAP50 (val):    {m_val.box.map50:.4f}")
    print(f"  > mAP (val):      {m_val.box.map75:.4f}")

    results = model(source='bdd10k/images/test', batch=16, project='runsbdd', name='bdd10k_test', stream=True)

    i = 0
    for result in results:
        print(f"  > {i} - {result.probs.cpu().numpy()}")
        result.save()
        i+=1
        if i > 10:
            break
    t3 = time.time()

if __name__ == '__main__':
    main()