from ultralytics import YOLO
#################################### COnstantes #####################################
YAML_PATH = 'd10k/bdd10k.yaml'


def main():

    print('Treinamento completo bdd10k')

    model = YOLO('yolo11n.pt')

    model.train(
        data=YAML_PATH,
        epochs=25,
        imgsz=640,
        batch=16,
        device=[0, 1],
        project='runsbdd',
        name='bdd10k',
        plots=True,
    )

    best_model_p = 'runsbdd/bdd10k/weights/best.pt'
    model = YOLO(best_model_p)

    m_val = model.val(
        data=YAML_PATH,
        split='val',
        name='bdd10k_val',
        project='runsbdd',
        save_json=True,
    )
    print(f"  > mAP50-95 (val): {m_val.box.map:.4f}")
    print(f"  > mAP50 (val):    {m_val.box.map50:.4f}")
    print(f"  > mAP (val):      {m_val.box.map75:.4f}")

    results = model(source='bdd10k/images/test', batch=16, project='runsbdd', name='bdd10k_test', stream=True)

    results_csv = results.to_csv()
    results_json = results.to_json()

    with open('runsbdd/bdd10k_test/bdd10k_test.csv', 'w') as f:
        f.write(results_csv)

    with open('runsbdd/bdd10k_test/bdd10k_test.json', 'w') as f:
        f.write(results_json)

    i = 0
    for result in results:
        print(f"  > {i} - {result.probs.cpu().numpy()}")
        result.save()
        i+=1
        if i > 10:
            break

if __name__ == '__main__':
    main()