import json
import re
import threading
import comet_ml
import argparse
from pathlib import Path
from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import DatasetType, DatasourcePurpose
from tqdm import tqdm
from ultralytics import YOLO
import os
from os import linesep
import shutil
from utils.move_files import move_folder
import torch
from typing import Dict, List
import pandas as pd
import time

######################################## CONSTANTES ########################################
LIGHTLY_TOKEN = "6ef4b5e20f6a1dba87a72a9eb4ddceb3f9529cd3d46b94a8" 
DATASET_PATH = Path('FOCAL/yolov5_format')
LIGHTLY_INPUT = Path('lightly')
ALL_IMAGES = Path('FOCAL/yolov5_format/images/all_images.txt')


def calculate_time(start: float, end: float) -> str:
    diff = end - start
    h = int(diff // 3600)
    m = int((diff % 3600) // 60)
    s = int(diff % 60)
    if h > 0:
        return f"{h} horas, {m} minutos e {s} segundos"
    elif m > 0:
        return f"{m} minutos e {s} segundos"
    else:
        return f"{s} segundos"

def create_dir(dataset_name:str) -> Path:

    if not os.path.exists(f'runs/{dataset_name}'):
        os.makedirs(f'runs/{dataset_name}')
        print(f'Diretório do dataset {dataset_name} criado com sucesso!')
    else:
        print(f'Esse diretório {dataset_name} já existe!')

    return Path(f'runs/{dataset_name}')


def configure_lightly_client(token: str, dataset_name: str) -> ApiWorkflowClient:
    """
    Configura o cliente Lightly com o token e o dataset.
    
    Returns:
        ApiWorkflowClient: Cliente configurado.
    """
    client = ApiWorkflowClient(token=token)
    try:
        client.create_dataset(
            dataset_name=dataset_name,
            dataset_type=DatasetType.IMAGES,
        )
    except Exception as e:
        print(f"Erro ao criar o dataset: {e}")
        # Se o dataset já existir, apenas configurar o cliente
        client.set_dataset_id_by_name(dataset_name=dataset_name)
        print(f"Usando o já existente.")

    client.set_local_config(purpose=DatasourcePurpose.INPUT)
    client.set_local_config(purpose=DatasourcePurpose.LIGHTLY)
    return client


def update_pool(
    client: ApiWorkflowClient,
    all_images_path: Path,
    cycle_name: str,
    run_dir: Path
) -> Dict[str, Path]:    
    """
    Atualiza o pool de imagens no Lightly.
    """
    print('='*80)
    print(f"\n--- Atualizando pool de dados para o ciclo: {cycle_name} ---")
    print('\n'+'='*80)
    
    # Define o diretório de configuração central para este ciclo
    config_dir = run_dir / "config" / cycle_name
    config_dir.mkdir(parents=True, exist_ok=True)
    pool_path = run_dir / "pool.csv"

    if not pool_path.exists():
        print('Criando o arquivo principal controle de imagens rotuladas e não rotuladas...')

        with open(all_images_path, 'r') as f:
            all_filenames = [line.strip() for line in f if line.strip()]
        
        df = pd.DataFrame(all_filenames, columns=['filename'])
        df['status'] = 0  # Status inicial para todas as imagens (não rotuladas)

    else:
        print('Arquivo de controle de imagens já existe. Atualizando...')
        df = pd.read_csv(pool_path)

    tags = client.get_all_tags()
    tags.sort(key=lambda t: t.created_at, reverse=True)

    new_tag = tags[0]
    image_names = client.export_filenames_by_tag_name(tag_name=new_tag.name).splitlines()


    new_labeled_images = []
    for name in image_names:
        if name:
            new_labeled_images.append(name.strip())

    # Atualiza o status para rotulado
    df.loc[df['filename'].isin(new_labeled_images), 'status'] = 1  

    df.to_csv(pool_path, index=False)

    abs_images_path = (DATASET_PATH / 'images' / 'train').absolute()
    
    df_rotulado = df[df['status'] == 1]
    df_nao_rotulado = df[df['status'] == 0]
    
    caminhos_rotulados = [str(abs_images_path / fname) for fname in df_rotulado['filename']]
    caminhos_nao_rotulados = [str(abs_images_path / fname) for fname in df_nao_rotulado['filename']]


    # Salva os arquivos .txt DENTRO do diretório de configuração do ciclo
    path_labeled_txt = config_dir / "labeled.txt"
    path_unlabeled_txt = config_dir / "unlabeled.txt"
    
    # Escreve os caminhos nos arquivos de texto
    print(f"Salvando lista de {len(caminhos_rotulados)} caminhos rotulados em: {path_labeled_txt}")
    with open(path_labeled_txt, "w") as f:
        f.write('\n'.join(caminhos_rotulados))

    print(f"Salvando lista de {len(caminhos_nao_rotulados)} caminhos não rotulados em: {path_unlabeled_txt}")
    with open(path_unlabeled_txt, "w") as f:
        f.write('\n'.join(caminhos_nao_rotulados))

    # Retorna os caminhos para os arquivos criados
    return {
        "labeled_txt_path": path_labeled_txt,
        "unlabeled_txt_path": path_unlabeled_txt
    }

def prepare_yolo_dataset(labeled_txt_path: Path) -> Path:
    """
    Cria o arquivo data.yaml para um ciclo específico, usando a lista de treino fornecida.
    Salva o .yaml na mesma pasta que o arquivo .txt.

    Returns:
        Path: O Path para o arquivo data.yaml criado.
    """
    # O diretório de configuração é o mesmo onde o 'labeled_txt_path' está
    config_dir = labeled_txt_path.parent

    print("=" * 100)
    print(f"Criando arquivo de configuração YAML em: {config_dir}")
    print(f"Arquivo de imagens rotuladas: {labeled_txt_path.absolute()} vai para o 'train' do data.yaml")
    print("=" * 100)
    
    yaml_path = config_dir / "data.yaml"
    yaml_content = f"""
# 'path' resolve os caminhos para 'val' e 'test'.
path: {DATASET_PATH.absolute()}

# Para 'train', usamos um caminho absoluto para evitar qualquer ambiguidade.
train: {labeled_txt_path.absolute()}

val: images/val
test: images/test

nc: 4
names:
  0: person
  1: bicycle
  2: car
  3: cart
""".strip()
    with open(yaml_path, "w") as f:
        f.write(yaml_content.strip())
    print(f"Arquivo de configuração YAML criado em: {yaml_path}")
    
    return yaml_path

#################################### Treinamento YOLO ####################################
def train_yolo(cycle_name:str, yaml_path:str, project_name:str, epochs:int, model_path='yolo11n.pt'):

    #Carregar modelo
    model = YOLO(model_path)

    #Treinar modelo
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        name=f"{cycle_name}",
        project=project_name,
        device=[0, 1],
        plots=True,
    )
    print("="*100)
    print(f"Treinamento concluído para o ciclo {cycle_name}. ")
    print("="*100)
    return results


def generate_lightly_predictions(model_path, image_paths: list[str], output_dir: Path, chunk_size: int):
    """Executa a predição, otimizado para o ambiente Kaggle."""
    if not image_paths:
        print("Nenhuma imagem para processar nesta partição.")
        return

    print(f"\nIniciando predição para {len(image_paths)} imagens...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Dividir as imagens igualmente para cada GPU
    mid = len(image_paths) // 2
    paths_splits = [image_paths[:mid], image_paths[mid:]]

    def worker_process(model_path, image_paths, output_dir, chunk_size, gpu_id, pbar, lock):
        """Processa as imagens em lotes e salva as predições."""
        model = YOLO(model_path)

        for i in range(0, len(image_paths), chunk_size):
            chunk_paths = image_paths[i:i + chunk_size]
            try:
                results = model(chunk_paths, stream=True, verbose=False, batch=32, device=gpu_id)
                for result in results:
                    original_filename = Path(result.path).name
                    output_json_path = output_dir / f"{Path(original_filename).stem}.json"
                    
                    if output_json_path.exists():
                        with lock:
                            pbar.update(1)
                        continue

                    lightly_data = {"file_name": original_filename, "predictions": []}
                    boxes = result.boxes.xywh.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    for j in range(len(boxes)):
                        lightly_data["predictions"].append({
                            "category_id": int(class_ids[j]),
                            "bbox": boxes[j].tolist(),
                            "score": float(confidences[j])
                        })
                    
                    with open(output_json_path, 'w') as f:
                        json.dump(lightly_data, f, indent=4)
                    
                    with lock:
                        pbar.update(1)
            except Exception as e:
                print(f"\nERRO ao processar o lote: {e}")
                time.sleep(2)
                with lock:
                    pbar.update(len(chunk_paths))
    
    with tqdm(total=len(image_paths), desc=f"Criando as predições das imagens") as pbar:

        lock = threading.Lock()
        threads = []

        for i, split in enumerate(paths_splits):

            thread = threading.Thread(
                target=worker_process,
                args=(model_path, split, output_dir, chunk_size, i, pbar, lock),
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()



def monitoring_run(client: ApiWorkflowClient, run_id: str):
    """
    Monitora múltiplos runs de workers no Lightly de forma concorrente.
    Args:
        client: A instância do cliente da API do Lightly.
        run_ids: Uma lista de strings, onde cada string é um ID de run agendado.
    """
    if not run_id:
        print("⚠️ Nenhuma ID de run foi fornecida.")
        return {}
    for run_info in client.compute_worker_run_info_generator(scheduled_run_id=run_id):
        print(f"Estado Atual: '{run_info.state}' ---> '{run_info.message}'")

    if run_info.ended_successfully():
        print(f"✅ Run [{run_id}] concluído com SUCESSO!")
    else:
        print(f"❌ Run [{run_id}] FALHOU com status final: '{run_info.state}'")


    print("\n🏁 Monitoramento concluído!")

def print_commands(dataset_path: Path, lightly_token: str):
    """
    Imprime os comandos necessários para executar o worker LightlyOne com o dataset.

    Args:
        dataset_path (Path): Caminho para o diretório do dataset.
        lightly_token (str): Token de autenticação do cliente Lightly.
    """
    gpus_flag = "--gpus all" if torch.cuda.is_available() else ""
    print(
        f"{linesep}Docker Run command: {linesep}"
        f"\033[7m"
        f"docker run {gpus_flag} --shm-size='32768m' --rm -it \\{linesep}"
        f"\t-v '{dataset_path.absolute()}/images/train':/input_mount:ro \\{linesep}"
        f"\t-v '{Path('lightly').absolute()}':/lightly_mount \\{linesep}"
        f"\t-e LIGHTLY_TOKEN={lightly_token} \\{linesep}"
        f"\tlightly/worker:latest{linesep}"
        f"\033[0m"
    )
    print(
        f"{linesep}Lightly Serve command:{linesep}"
        f"\033[7m"
        f"lightly-serve \\{linesep}"
        f"\tinput_mount='{dataset_path.absolute()}/images/train' \\{linesep}"
        f"\tlightly_mount='{Path('lightly').absolute()}'{linesep}"
        f"\033[0m"
    )


def evaluate_yolo(model_path, yaml_path: Path, output_dir: Path, name: str, project: str):
    """
    Avalia o modelo YOLO com os conjuntos de validação e teste.
    
    Args:
        model_path (str): Caminho para o modelo YOLO treinado.
        yaml_path (str): Caminho para o arquivo de configuração do dataset.
        output_dir (Path): Diretório onde os resultados serão salvos.
    """
    model = YOLO(model_path)

    # Avaliar no conjunto de validação
    metrics_val = model.val(data=yaml_path, split='val', name=f'{name}_val', project=project)
    print(f"  > mAP50-95 (val): {metrics_val.box.map:.4f}")
    print(f"  > mAP50 (val):    {metrics_val.box.map50:.4f}")


    val_csv = metrics_val.to_csv()
    csv_filename = output_dir / "validation_results.csv"
    with open(csv_filename, "w") as f:
        f.write(val_csv)

    # Avaliar no conjunto de teste
    metrics_test = model.val(data=yaml_path, split='test', name=f'{name}_test', project=project)

    print(f"  > mAP50-95 (test): {metrics_test.box.map:.4f}")
    print(f"  > mAP50 (test):    {metrics_test.box.map50:.4f}")
    
    test_csv = metrics_test.to_csv()
    csv_filename = output_dir / "test_results.csv"
    with open(csv_filename, "w") as f:
        f.write(test_csv)


def main(dataset_name: str, epochs: int, initial_model_path: str = 'yolo11n.pt', start: int = 0, selection_type: str = 'uncert', retrain: bool = False):

    if selection_type == 'uncert':
        print('='*80)
        print(f"Configuração de seleção: sem balanceamento")
        print('='*80)
        selection_config = {
            "proportion_samples": 0.01, # +1% do dataset
            "strategies": [
                {
                    "input": {
                        "type": "SCORES",
                        "task": "object_detection", 
                        "score": "uncertainty_entropy",
                    },
                    "strategy": {
                        "type": "WEIGHTS"
                    }
                },
                {
                    "input": {
                        "type": "EMBEDDINGS",
                    },
                    "strategy": {
                        "type": "DIVERSITY",
                    }
                },
            ],
        }
    elif selection_type == 'balance':
        print('='*80)
        print(f"Configuração de seleção: com balanceamento")
        print('='*80)

        selection_config = {
            "proportion_samples": 0.01, # +1% do dataset
            "strategies": [
                {
                    "input": {
                        "type": "SCORES",
                        "task": "object_detection", 
                        "score": "uncertainty_entropy",
                    },
                    "strategy": {
                        "type": "WEIGHTS"
                    }
                },
                {
                    "input": {
                        "type": "EMBEDDINGS",
                    },
                    "strategy": {
                        "type": "DIVERSITY",
                    },
                },
                {
                    "input": {
                        "type": "PREDICTIONS",
                        "task": "object_detection",
                        "name": "CLASS_DISTRIBUTION"
                    },
                    "strategy": {  
                        "type": "BALANCE",  
                        "distribution": "UNIFORM"  
                    }
                }                
            ],
        }
        

    comet_ml.login(project_name=dataset_name)

    #################### 1. criar o diretório de saída da execução ####################
    run_dir = create_dir(dataset_name)

    #################### 2. iniciar o cliente do lightly ####################
    client = configure_lightly_client(LIGHTLY_TOKEN, dataset_name)

    baseline_model = Path(dataset_name) / "ciclo_0" / "weights" / "best.pt"

    num_total_cycles = 10

    for i in range(start, num_total_cycles):

        print('='*100)
        print(f"--- Iniciando o ciclo {i} ---")
        print('='*100)
        cycle_name = f'ciclo_{i}'

        if i == 0:            
            t1 = time.time()
            scheduled_run_id = client.schedule_compute_worker_run(
                worker_config = {
                    "shutdown_when_job_finished": True,
                    "use_datapool": True,
                    "datasource": {
                        "process_all": True,
                    },
                },
                selection_config={
                    "proportion_samples": 0.01, # 1% do dataset
                    "strategies": [
                        {
                            "input": {
                                "type": "RANDOM",
                                "random_seed": 42, # optional, for reproducibility
                            },
                            "strategy": {
                                "type": "WEIGHTS",
                            }
                        }
                    ]
                },
            )
            print(f'Executando o worker LightlyOne para selecionar aleatoriamente 1% do dataset.')
        else:
            t1 = time.time()
            scheduled_run_id = client.schedule_compute_worker_run(
                worker_config = {
                    "shutdown_when_job_finished": True,
                    "use_datapool": True,
                    "datasource": {
                        "process_all": True,
                    },
                },
                selection_config=selection_config
            )
            print(f'Executando o worker LightlyOne para selecionar 1% do dataset.')
        ####################################### Seleção das imagens ########################################
        print('\n\n')
        print_commands(DATASET_PATH, LIGHTLY_TOKEN)
        monitoring_run(client, scheduled_run_id)
        t2 = time.time()
        print(f"Tempo total de execução da seleção no ciclo {i}: {calculate_time(t1, t2)}")
        ####################################################################################################
        ################################ Atualização do pool de dados ######################################
        data_splits = update_pool(client, ALL_IMAGES, cycle_name, run_dir)
        labeled_txt = data_splits["labeled_txt_path"]
        unlabeled_txt = data_splits["unlabeled_txt_path"]
        yaml_path = prepare_yolo_dataset(labeled_txt)
        t3 = time.time()
        print(f"Tempo total da atualização do pool no ciclo {i}: {calculate_time(t2, t3)}")
        ####################################################################################################
        #################################### Escolha do modelo inicial #####################################
        if i == 0:
            init_model_path = initial_model_path
        else:
            if retrain:
                init_model_path = str(baseline_model.absolute())
            else:
                init_model_path = str((Path(dataset_name) / f'ciclo_{i-1}' / "weights" / "best.pt").absolute())
        print(f"Modelo inicial para o ciclo {i}: {init_model_path}")    
        ####################################################################################################
        #################################### Treinamento do modelo YOLO ####################################
        results = train_yolo(cycle_name, str(yaml_path.absolute()), dataset_name, epochs=epochs, model_path=init_model_path)
        print(f"Resultados do ciclo {cycle_name}: {results}")
        t4 = time.time()
        print(f"Tempo total do treinamento no ciclo {i}: {calculate_time(t3, t4)}")
        ####################################################################################################
        ##################################### Avaliação do modelo YOLO #####################################
        final_model_path = str((Path(dataset_name) / cycle_name / "weights" / "best.pt").absolute())

        evaluate_yolo(
            model_path=final_model_path,
            yaml_path=str(yaml_path.absolute()),
            output_dir=labeled_txt.parent,
            name=cycle_name,
            project=dataset_name
        )
        t5 = time.time()
        print(f"Tempo total da avaliação do modelo no ciclo {i}: {calculate_time(t4, t5)}")
        ####################################################################################################
        ############################## Geração de predições para o LightlyOne ##############################
        if i < num_total_cycles - 1:
        # Lê a lista de caminhos não rotulados do arquivo de texto
            with open(unlabeled_txt, 'r') as f:
                image_paths_for_prediction = [line.strip() for line in f if line.strip()]
            
            generate_lightly_predictions(
                model_path=final_model_path, # Carrega o modelo treinado
                image_paths=image_paths_for_prediction,
                output_dir=LIGHTLY_INPUT / '.lightly' / 'predictions' / 'object_detection',
                chunk_size=64
            )
            t6 = time.time()
            print(f"Tempo total da geração de predições no ciclo {i}: {calculate_time(t5, t6)}")
        ####################################################################################################
        print(f"Tempo total de execução do ciclo {i}: {calculate_time(t1, t6)}")

    move_folder(dataset_name, f'runs/{dataset_name}')
    

def complete_train(dataset_name: str, epochs: int):

    run_dir = create_dir('complete_train');

    data_yaml = prepare_yolo_dataset(ALL_IMAGES)

    # 2. Ler o conteúdo do arquivo gerado
    with open(data_yaml, "r") as f:
        content = f.read()

    # 3. Substituir a linha 'train:' usando uma expressão regular para garantir
    #    que apenas essa linha específica seja alterada.
    #    O padrão `^train: .*` casa com qualquer linha que comece com "train:"
    modified_content = re.sub(
        pattern=r"^train: .*", 
        repl="train: images/train", 
        string=content, 
        flags=re.MULTILINE
    )

    print(modified_content)

    with open(data_yaml, "w") as f:
        f.write(modified_content)

    #################################################################################################### Total 4 classes 
    model_1 = YOLO('yolo11n.pt')

    print("="*100)
    print('Treinamento completo do modelo YOLO, usando todas as imagens rotuladas e as 4 classes.')
    print("="*100)

    model_1.train(
        data=str(data_yaml.absolute()),
        epochs=epochs,
        project=dataset_name,
        name='classes_4',
        device=[0, 1],
        plots=True,
    )

    f_m_path = str((Path(dataset_name)  / "classes_4" / "weights" / "best.pt").absolute())

    evaluate_yolo(
        model_path=f_m_path,
        yaml_path=str(data_yaml.absolute()),
        output_dir=Path(dataset_name),
        name='classes_4',
        project=dataset_name
    )

    print(f"Treinamento e avaliação concluídos.")
    time.sleep(10)
    print("="*100)

    #################################################################################################### Total 3 classes

    print("="*100)
    print('Treinamento completo do modelo YOLO, usando todas as imagens rotuladas e as 3 primeiras classes.')
    print("="*100)

    model_2 = YOLO('yolo11n.pt')
    model_2.train(
        data=str(data_yaml.absolute()),
        epochs=epochs,
        project=dataset_name,
        name='classes_3',
        device=[0, 1],
        plots=True,
        classes=[0, 1, 2]  # Classes: person, bicycle, car
    )

    final_model_path = str((Path(dataset_name)  / "classes_3" / "weights" / "best.pt").absolute())

    final_model = YOLO(final_model_path)
    res_val = final_model.val(data=str(data_yaml.absolute()), split='val', name='classes_3_val', project=dataset_name)
    res_val_csv = res_val.to_csv()
    csv_filename = Path(dataset_name) / "classes_3" / "val_results.csv"
    with open(csv_filename, "w") as f:
        f.write(res_val_csv)

    res_test = final_model.val(data=str(data_yaml.absolute()), split='test', name='classes_3_test', project=dataset_name)
    res_test_csv = res_test.to_csv()
    csv_filename = Path(dataset_name) / "classes_3" / "test_results.csv"
    with open(csv_filename, "w") as f:
        f.write(res_test_csv)
        
    
    print(f"Treinamento e avaliação concluídos.")
    time.sleep(10)
    print("="*100)
    ####################################################################################################
    move_folder(dataset_name, f'runs/complete_train')

    print(f"{dataset_name} concluído e movido para o diretório 'runs/complete_train'.")

    #################################################################################################### COM FREEZE E 4 CLASSES

    print("="*100)
    print (f'Treinamento do modelo para 1% aleatório do dataset COM freeze e com 4 classes.')
    print("="*100)
    model_3 = YOLO('yolo11n.pt')
    yaml_3 = Path('runs/pipeline2/config/ciclo_0/data.yaml') 
    model_3.train(
        data=str(yaml_3.absolute()),
        epochs=epochs,
        project='freeze_4_classes',
        name='classes_4',
        device=[0, 1],
        plots=True,
        freeze=10
    )

    f_model_3 = str((Path('freeze_4_classes')  / "classes_4" / "weights" / "best.pt").absolute())

    evaluate_yolo(
        model_path=f_model_3,
        yaml_path=str(yaml_3.absolute()),
        output_dir=Path('freeze_4_classes'),
        name='classes_4',
        project='freeze_4_classes'
    )
    print(f"Treinamento e avaliação concluídos.")
    time.sleep(10)
    print("="*100)
    ###################################################################################################
    print("="*100)
    print (f'Treinamento do modelo para 1% aleatório do dataset COM freeze, com 4 classes. e learning rate de 0.0001.')
    print("="*100)

    model_4 = YOLO('yolo11n.pt')
    yaml_4 = Path('runs/pipeline2/config/ciclo_0/data.yaml')
    model_4.train(
        data=str(yaml_4.absolute()),
        epochs=epochs,
        project='freeze_4_classes',
        name='lr_1e4',
        device=[0, 1],
        plots=True,
        freeze=10,
        lr0=0.0001,
    )
    f_model_4 = str((Path('freeze_4_classes')  / "lr_1e4" / "weights" / "best.pt").absolute())

    evaluate_yolo(
        model_path=f_model_4,
        yaml_path=str(yaml_4.absolute()),
        output_dir=Path('freeze_4_classes'),
        name='lr_1e4',
        project='freeze_4_classes'
    )
    print(f"Treinamento e avaliação concluídos.")
    time.sleep(10)
    print("="*100)

    move_folder('freeze_4_classes', f'runs/complete_train')

    ####################################################################################################

    print("="*100)
    print (f'Treinamento do modelo para 1% aleatório do dataset sem freeze e com 3 classes.')
    print("="*100)
    model_5 = YOLO('yolo11n.pt')
    yaml_5 = Path('runs/pipeline2/config/ciclo_0/data.yaml') 
    model_5.train(
        data=str(yaml_5.absolute()),
        epochs=epochs,
        project='freeze_3_classes',
        name='classes_3_sem_freeze',
        device=[0, 1],
        plots=True,
        classes=[0, 1, 2]  # Classes: person, bicycle, car
    )


    print(f"Treinamento e avaliação concluídos.")
    time.sleep(10)
    print("="*100)
    ###############################################################################################
    print("="*100)
    print (f'Treinamento do modelo para 1% aleatório do dataset COM freeze e com 3 classes.')
    print("="*100)

    model_6 = YOLO('yolo11n.pt')
    yaml_6 = Path('runs/pipeline2/config/ciclo_0/data.yaml')
    model_6.train(
        data=str(yaml_6.absolute()),
        epochs=epochs,
        project='freeze_3_classes',
        name='classes_3',
        device=[0, 1],
        plots=True,
        classes=[0, 1, 2],
        freeze=10
    )

    print(f"Treinamento e avaliação concluídos.")
    time.sleep(10)
    print("="*100)
   #######################################################################################################################
    print("="*100)
    print (f'Treinamento do modelo para 1% aleatório do dataset COM freeze, com 3 classes. e learning rate de 0.0001.')
    print("="*100)

    model_7 = YOLO('yolo11n.pt')
    yaml_7 = Path('runs/pipeline2/config/ciclo_0/data.yaml')
    model_7.train(
        data=str(yaml_7.absolute()),
        epochs=epochs,
        project='freeze_3_classes',
        name='lr_1e4',
        device=[0, 1],
        plots=True,
        freeze=10,
        classes=[0, 1, 2],  # Classes: person, bicycle, car
        lr0=0.0001,  
    )

    print(f"Treinamento e avaliação concluídos.")
    time.sleep(10)
    print("="*100)

    move_folder('freeze_3_classes', f'runs/complete_train')


def validate_yolo_zero(name:str):
    """
    Valida a yolo sem treinamento com o conjunto de validação
    """
    create_dir('yolo11n_zero_validation')
    data_yaml = prepare_yolo_dataset(ALL_IMAGES)
    model = YOLO('yolo11n.pt')

    results = model.val(
        data=str(data_yaml.absolute()),
        project='yolo11n_zero_validation',
        name=name,
        split='val',
        device=[0, 1],
        plots=True,
        classes = [0, 1, 2]  # Classes: person, bicycle, car, cart
    )
    val_csv = results.to_csv()
    csv_filename = Path('yolo11n_zero_validation') / f'{name}' / "validation_results.csv"
    with open(csv_filename, "w") as f:
        f.write(val_csv)

    move_folder('yolo11n_zero_validation', f'runs/yolo11n_zero_validation')


if __name__ == "__main__":

    parse = argparse.ArgumentParser(description="Train YOLO model with Lightly dataset")
    parse.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset")
    parse.add_argument("-e", "--epochs", type=int, default=25, help="Number of training epochs")
    parse.add_argument("-m", "--model", type=str, default='yolo11n.pt', help="Path to the YOLO model to train from")
    parse.add_argument("-s", "--start", type=int, default=0, help="Starting cycle number")
    parse.add_argument("-t", "--type", type=str, default='uncert', help="Type of selection strategy to use (uncertainty with or without balance)")
    parse.add_argument("-r", "--retrain", action='store_true', help="Retrain the model from the beginning")
    parse.add_argument("--mode", type=str, default='al', choices=['al', 'val', 'train'], help="Mode of operation: 'al' for active learning, 'val' for zero validation, 'train' for complete training")
    parse.add_argument("--debug", action='store_true', help="Enable debug mode")

    args = parse.parse_args()

    dataset_name = args.dataset
    epochs = args.epochs
    start = args.start
    selection_type = args.type

    if args.debug:
        print('='*100)
        print(f"Debug mode is ON. Dataset: {dataset_name}, Epochs: {epochs}, Start Cycle: {start}, Selection Type: {selection_type}")
        print(f"Initial Model Path: {args.model}, Retrain: {args.retrain}")
        print('='*100)

        # complete_train(dataset_name, epochs)

        init_model = str((Path(dataset_name) / f'ciclo_{start}' / "weights" / "best.pt").absolute())
        t1 = time.time()

        # model = YOLO('yolo11n.pt')

        # results = model.train(
        #     data=str(Path('data.yaml').absolute()),
        #     epochs=epochs,
        #     name=f'ciclo_{start}',
        #     project=f'debug_train',
        #     device=[0, 1],
        #     plots=True,
        # )

        train_yolo(
            cycle_name=f'ciclo_{start}',
            yaml_path=str(Path(f'runs/{dataset_name}/config/ciclo_{start}/data.yaml').absolute()),
            project_name=f'debug_{dataset_name}',
            epochs=epochs,
            model_path=init_model
        )
        t2 = time.time()
        print(f"Tempo total do treinamento no ciclo {start}: {calculate_time(t1, t2)}")


        final_model = Path(f'debug_{dataset_name}/ciclo_{start}/weights/best.pt')

        txt_path = Path(f'runs/{dataset_name}/config/ciclo_{start}/ciclo_{start}_unlabeled.txt')


        with open(txt_path, 'r') as f:
            image_paths_for_prediction = [line.strip() for line in f if line.strip()]
        t3 = time.time()
        generate_lightly_predictions(
            model_path=str(final_model.absolute()),
            image_paths=image_paths_for_prediction,
            output_dir=Path('runs') / f'debug_{dataset_name}'/ 'predictions',
            chunk_size=64
        )
        t4 = time.time()
        print(f"Tempo total da geração de predições no ciclo {start}: {calculate_time(t3, t4)}")

        print(f"Tempo total de execução do ciclo {start}: {calculate_time(t1, t4)}")
    
    elif args.mode == 'val':
        
        validate_yolo_zero(name=dataset_name)
    elif args.mode == 'train':
        print('='*100)
        print(f"Modo de operação: treinamento completo do modelo YOLO")
        print('='*100)

        complete_train(dataset_name, epochs)

    elif args.mode == 'al':
        ts = time.time()
        main(
            dataset_name=dataset_name,
            epochs=epochs,
            initial_model_path=args.model,
            start=start,
            selection_type=selection_type,
            retrain=args.retrain
        )
        te = time.time()
        print(f"Tempo total de execução do pipeline: {calculate_time(ts, te)}")
