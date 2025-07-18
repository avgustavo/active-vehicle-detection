# src/run_baseline_experiment.py

import os
import yaml
import random
import shutil
from pathlib import Path
from ultralytics import YOLO

# --- CONFIGURAÇÕES DO EXPERIMENTO ---
PROJECT_ROOT = Path(__file__).parent.parent
DATASET_YOLO_PATH = PROJECT_ROOT / "dataset_yolo"
RESULTS_PATH = PROJECT_ROOT / "results"
EXPERIMENT_NAME = "baseline_vehicle_detection" # ALTERAÇÃO: Nome do experimento atualizado

# Parâmetros do Ciclo de Treinamento
INITIAL_SAMPLE_SIZE = 100
SAMPLES_PER_CYCLE = 100
NUM_CYCLES = 10
YOLO_MODEL = 'yolov8n.pt'
EPOCHS = 25
IMG_SIZE = 640

def create_dataset_yaml(train_image_list_path, val_path, class_names, output_path):
    """Cria o arquivo de configuração .yaml para o YOLO."""
    config = {
        'train': str(train_image_list_path.resolve()),
        'val': str(val_path.resolve()),
        # ALTERAÇÃO: nc (number of classes) é inferido pelo YOLO a partir dos nomes
        'names': {i: name for i, name in enumerate(class_names)}
    }
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Arquivo de configuração do dataset salvo em: {output_path}")

def main():
    """Orquestra o experimento de baseline com amostragem aleatória."""
    exp_dir = RESULTS_PATH / EXPERIMENT_NAME
    if exp_dir.exists():
        print(f"Diretório de experimento {exp_dir} já existe. Removendo para um novo run.")
        shutil.rmtree(exp_dir)
    exp_dir.mkdir(parents=True)
    
    train_images_path = DATASET_YOLO_PATH / "images" / "train"
    full_training_pool = [p for p in train_images_path.glob("*.png")]
    random.shuffle(full_training_pool)
    
    print(f"Encontrado um total de {len(full_training_pool)} imagens no pool de treino.")
    
    labeled_set = []
    
    for cycle in range(NUM_CYCLES):
        print("\n" + "="*50)
        print(f"INICIANDO CICLO {cycle+1}/{NUM_CYCLES} - DETECÇÃO DE VEÍCULOS")
        print("="*50)
        
        num_samples_to_add = INITIAL_SAMPLE_SIZE if cycle == 0 else SAMPLES_PER_CYCLE
        new_samples = full_training_pool[:num_samples_to_add]
        full_training_pool = full_training_pool[num_samples_to_add:]
        
        labeled_set.extend(new_samples)
        
        print(f"Adicionando {len(new_samples)} novas amostras. Total de dados rotulados: {len(labeled_set)}")
        
        cycle_train_list_path = exp_dir / f"train_cycle_{cycle+1}.txt"
        with open(cycle_train_list_path, 'w') as f:
            for img_path in labeled_set:
                f.write(str(img_path.resolve()) + "\n")
                
        cycle_yaml_path = exp_dir / f"dataset_cycle_{cycle+1}.yaml"
        create_dataset_yaml(
            train_image_list_path=cycle_train_list_path,
            val_path=DATASET_YOLO_PATH / "images" / "val",
            # ALTERAÇÃO: Nossas duas classes aqui!
            class_names=["car", "motorcycle"],
            output_path=cycle_yaml_path
        )
        
        print(f"\nIniciando treino do YOLO para o ciclo {cycle+1}...")
        model = YOLO(YOLO_MODEL)
        model.train(
            data=str(cycle_yaml_path.resolve()),
            epochs=EPOCHS,
            imgsz=IMG_SIZE,
            project=str(RESULTS_PATH.resolve()),
            name=f"{EXPERIMENT_NAME}/cycle_{cycle+1}",
            exist_ok=True
        )
        
        print(f"Ciclo {cycle+1} concluído. Resultados salvos em {RESULTS_PATH}/{EXPERIMENT_NAME}/cycle_{cycle+1}")

    print("\n\nExperimento de baseline (detecção de veículos) concluído!")

if __name__ == "__main__":
    main()