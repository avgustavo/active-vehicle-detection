import os
import yaml
import random
import shutil
from pathlib import Path
from ultralytics import YOLO
from datetime import datetime

# --- CONFIGURAÇÕES DO EXPERIMENTO ---

# 1. Caminho para a pasta de dados processados pelo script anterior
PROCESSED_DATA_DIR = Path("./yolo_dataset")

# 2. Defina o "budget" de anotação por ciclo
K_SAMPLES_PER_CYCLE = 200 # Usaremos um valor menor para ver a progressão mais rápido

# 3. Número total de ciclos a executar
TOTAL_CYCLES = 9

# 4. Configurações do Treinamento YOLO
YOLO_MODEL_START_WEIGHTS = 'yolov8n.pt'  # Modelo pré-treinado (nano é o mais rápido)
TRAINING_EPOCHS_PER_CYCLE = 10          # Número de épocas para treinar a cada ciclo
IMAGE_SIZE = 640

# 5. Nome do projeto para organizar os resultados dos treinos
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
EXPERIMENT_NAME = f"baseline_random_sampling_{TIMESTAMP}"


# --- SCRIPT PRINCIPAL ---

def main():
    """
    Orquestra o experimento de baseline com amostragem aleatória,
    treinando um modelo YOLO a cada ciclo.
    """
    # --- 1. PREPARAÇÃO INICIAL ---
    
    # Caminhos para os dados de treino e validação
    train_images_dir = PROCESSED_DATA_DIR / "train" / "images"
    original_dataset_yaml = PROCESSED_DATA_DIR / "dataset.yaml"

    # Criar um diretório temporário para os arquivos de dados dos ciclos
    temp_data_dir = Path("./temp_data")
    temp_data_dir.mkdir(exist_ok=True)

    # Carregar todo o pool de imagens de treinamento disponíveis
    print("Mapeando o pool de dados de treinamento...")
    unlabeled_pool = sorted(list(train_images_dir.glob("*.png")))
    print(f"Encontradas {len(unlabeled_pool)} imagens no pool total de treinamento.")
    
    # Inicializar listas para o experimento
    labeled_set_paths = []
    results_history = []

    print("\n" + "="*60)
    print("INICIANDO EXPERIMENTO DE TREINAMENTO BASELINE (AMOSTRAGEM ALEATÓRIA)")
    print("="*60 + "\n")

    # --- 2. LOOP DE TREINAMENTO ITERATIVO ---

    for cycle in range(1, TOTAL_CYCLES + 1):
        print(f"\n--- CICLO DE TREINAMENTO {cycle}/{TOTAL_CYCLES} ---")
        
        # Verificar se há amostras suficientes no pool
        if len(unlabeled_pool) < K_SAMPLES_PER_CYCLE:
            print("Pool de dados insuficiente para o próximo ciclo. Encerrando.")
            break

        # a. Selecionar 'k' amostras aleatoriamente do pool
        newly_selected_paths = random.sample(unlabeled_pool, K_SAMPLES_PER_CYCLE)
        
        # b. "Anotar" as amostras: movê-las do pool para o conjunto rotulado
        for path in newly_selected_paths:
            unlabeled_pool.remove(path)
        labeled_set_paths.extend(newly_selected_paths)
        labeled_set_paths.sort() # Manter uma ordem consistente
        
        print(f"Amostragem Aleatória: {len(newly_selected_paths)} novas imagens selecionadas.")
        print(f"Tamanho total do conjunto de treinamento para este ciclo: {len(labeled_set_paths)} imagens.")

        # c. Preparar os arquivos de configuração para o YOLO neste ciclo
        
        # Criar um arquivo .txt listando todas as imagens no conjunto rotulado atual
        cycle_train_list_path = temp_data_dir / f"train_cycle_{cycle}.txt"
        with open(cycle_train_list_path, 'w') as f:
            for img_path in labeled_set_paths:
                f.write(f"{img_path.resolve()}\n")
        
        # Criar um arquivo .yaml customizado para este ciclo
        with open(original_dataset_yaml, 'r') as f:
            data_config = yaml.safe_load(f)
            
        data_config['train'] = str(cycle_train_list_path.resolve())
        # O caminho de validação permanece o mesmo sempre
        data_config['val'] = str((PROCESSED_DATA_DIR / data_config['val']).resolve())
        data_config.pop('test', None) # Remove a chave de teste se existir
        data_config.pop('path', None) # Remove a chave de path para usar caminhos absolutos
        
        cycle_yaml_path = temp_data_dir / f"data_cycle_{cycle}.yaml"
        with open(cycle_yaml_path, 'w') as f:
            yaml.dump(data_config, f, sort_keys=False)

        # d. Treinar o modelo YOLO com o subconjunto de dados atual
        print("\nIniciando o treinamento YOLO para o ciclo atual...")
        
        # Carregar o modelo YOLO com os pesos pré-treinados
        model = YOLO(YOLO_MODEL_START_WEIGHTS)

        # Treinar o modelo
        results = model.train(
            data=str(cycle_yaml_path.resolve()),
            epochs=TRAINING_EPOCHS_PER_CYCLE,
            imgsz=IMAGE_SIZE,
            project=EXPERIMENT_NAME,
            name=f"cycle_{cycle}_samples_{len(labeled_set_paths)}",
            exist_ok=True, # Permite sobrescrever treinos anteriores se o nome for o mesmo
            patience=5, # Early stopping
            verbose=False # Deixe como True se quiser ver todo o output do YOLO
        )
        
        print("Treinamento concluído.")
        
        # e. Salvar e imprimir os resultados de validação
        # A biblioteca ultralytics já roda a validação no final do treino
        # Usamos o mAP50 (mean Average Precision @ IoU=0.50) que é uma métrica comum.
        map50 = results.metrics.box.map50
        
        print(f"\nResultados do Ciclo {cycle}:")
        print(f"\tTotal de Amostras Anotadas: {len(labeled_set_paths)}")
        print(f"\tPerformance (mAP@.50): {map50:.4f}")

        results_history.append({
            "cycle": cycle,
            "labeled_count": len(labeled_set_paths),
            "mAP50": f"{map50:.4f}"
        })

    # --- 3. EXIBIR RESULTADOS FINAIS ---
    
    print("\n" + "="*60)
    print("EXPERIMENTO DE BASELINE (AMOSTRAGEM ALEATÓRIA) CONCLUÍDO")
    print("="*60 + "\n")
    print("Histórico de Performance:")
    print("Ciclo | Amostras Anotadas | Performance (mAP@.50)")
    print("------|-------------------|-----------------------")
    for res in results_history:
        print(f"{res['cycle']:<5} | {res['labeled_count']:<17} | {res['mAP50']}")

    # Opcional: limpar pasta temporária
    # shutil.rmtree(temp_data_dir)
    print(f"\nArquivos temporários de configuração salvos em: '{temp_data_dir}'")
    print(f"Resultados de todos os treinos salvos em: '{Path(EXPERIMENT_NAME)}'")


if __name__ == "__main__":
    main()