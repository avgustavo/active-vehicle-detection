import json
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
import time
from typing import Dict, List
import pandas as pd


######################################## CONSTANTES ########################################
LIGHTLY_TOKEN = "6ef4b5e20f6a1dba87a72a9eb4ddceb3f9529cd3d46b94a8" 
DATASET_PATH = Path('FOCAL/yolov5_format')
LIGHTLY_INPUT = Path('lightly')
DATA_POOL = Path('pool.csv')
ALL_IMAGES = Path('FOCAL/yolov5_format/images/all_images.txt')



def create_dir(dataset_name):

    if not os.path.exists(f'runs/{dataset_name}'):
        os.makedirs(f'runs/{dataset_name}')
        print(f'Diretório do dataset {dataset_name} criado com sucesso!')
    else:
        print(f'Esse diretório já existe!')


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

    client.set_local_config(purpose=DatasourcePurpose.INPUT)
    client.set_local_config(purpose=DatasourcePurpose.LIGHTLY)
    return client


def update_pool(
    client: ApiWorkflowClient,
    pool_path: Path,
    all_images_path: Path,
    cycle_name: str,
    project_name: str
) -> Dict[str, Path]:    
    """
    Atualiza o pool de imagens no Lightly.
    """
    print('='*80)
    print(f"\n--- Atualizando pool de dados para o ciclo: {cycle_name} ---")
    
    # Define o diretório de configuração central para este ciclo
    config_dir = Path("runs") / project_name / "config" / cycle_name
    config_dir.mkdir(parents=True, exist_ok=True)

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
    path_labeled_txt = config_dir / f"{cycle_name}_labeled.txt"
    path_unlabeled_txt = config_dir / f"{cycle_name}_unlabeled.txt"
    
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
  0: pedestrian
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
    
    return results


def generate_lightly_predictions(model_path, image_paths: list[str], output_dir: Path, batch_size: int):
    """Executa a predição, otimizado para o ambiente Kaggle."""
    if not image_paths:
        print("Nenhuma imagem para processar nesta partição.")
        return

    model = YOLO(model_path)
    print(f"\nIniciando predição para {len(image_paths)} imagens...")
    
    with tqdm(total=len(image_paths), desc=f"Criando as predições das imagens") as pbar:
        for i in range(0, len(image_paths), batch_size):
            chunk_paths = image_paths[i:i + batch_size]
            try:
                results = model(chunk_paths, stream=True, verbose=False, device=[0, 1])
                for result in results:
                    original_filename = Path(result.path).name
                    output_json_path = output_dir / f"{Path(original_filename).stem}.json"
                    
                    if output_json_path.exists():
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
                    
                    pbar.update(1)
            except Exception as e:
                print(f"\nERRO ao processar o lote: {e}")
                time.sleep(2)
                pbar.update(len(chunk_paths))


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


def main():
    parse = argparse.ArgumentParser(description="Train YOLO model with Lightly dataset")
    parse.add_argument("-d", "--dataset", type=str, required=True, help="Name of the dataset")
    parse.add_argument("-e", "--epochs", type=int, default=25, help="Number of training epochs")
    parse.add_argument("-m", "--model", type=str, default='yolo11n.pt', help="Path to the YOLO model to train from")

    args = parse.parse_args()

    dataset_name = args.dataset
    epochs = args.epochs

    comet_ml.login(project_name=dataset_name)

    #################### 1. criar o diretório de saída da execução ####################
    create_dir(dataset_name)

    #################### 2. iniciar o cliente do lightly ####################
    client = configure_lightly_client(LIGHTLY_TOKEN, dataset_name)

    baseline_model = Path(dataset_name) / "ciclo_0" / "weights" / "best.pt"

    num_total_cycles = 10

    for i in range(num_total_cycles):

        if i == 0:
            cycle_name = "ciclo_0"
            # scheduled_run_id = client.schedule_compute_worker_run(
            #     worker_config = {
            #         "shutdown_when_job_finished": True,
            #         "use_datapool": True,
            #         "datasource": {
            #             "process_all": True,
            #         },
            #     },
            #     selection_config={
            #         "proportion_samples": 0.01, # 1% do dataset
            #         "strategies": [
            #             {
            #                 "input": {
            #                     "type": "RANDOM",
            #                     "random_seed": 42, # optional, for reproducibility
            #                 },
            #                 "strategy": {
            #                     "type": "WEIGHTS",
            #                 }
            #             }
            #         ]
            #     },
            # )
            # print(f'Executando o worker LightlyOne para selecionar aleatoriamente 1% do dataset.')
            # print('\n\n')
            # print_commands(DATASET_PATH, LIGHTLY_TOKEN)
            
            # monitoring_run(client, scheduled_run_id)
            # 4. criar o arquivo de tags com os caminhos das imagens selecionadas
            data_splits = update_pool(client, DATA_POOL, ALL_IMAGES, cycle_name, dataset_name)

            labeled_txt = data_splits["labeled_txt_path"]
            unlabeled_txt = data_splits["unlabeled_txt_path"]
            
            # 5. criar o arquivo yaml de configuração do modelo
            yaml_path = prepare_yolo_dataset(labeled_txt)

            results = train_yolo(cycle_name, str(yaml_path.absolute()), dataset_name, epochs=epochs, model_path=args.model)

            print(f"Resultados do ciclo {cycle_name}: {results}")

            model = baseline_model

            evaluate_yolo(
                model_path=str(model.absolute()),
                yaml_path=str(yaml_path.absolute()),
                output_dir=labeled_txt.parent,
                name=cycle_name,
                project=dataset_name
            )

        else:
            cycle_name = f'ciclo_{i}'
            scheduled_run_id = client.schedule_compute_worker_run(
                worker_config = {
                    "shutdown_when_job_finished": True,
                    "use_datapool": True,
                    "datasource": {
                        "process_all": True,
                    },
                },
                selection_config={
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
                            }
                        ],
                    },
            )
            print(f'Executando o worker LightlyOne para selecionar 1% do dataset.')
            print('\n\n')
            print_commands(DATASET_PATH, LIGHTLY_TOKEN)
            
            monitoring_run(client, scheduled_run_id)

            data_splits = update_pool(client, DATA_POOL, ALL_IMAGES, cycle_name, dataset_name)

            labeled_txt = data_splits["labeled_txt_path"]
            unlabeled_txt = data_splits["unlabeled_txt_path"]

            yaml_path = prepare_yolo_dataset(labeled_txt)


            results = train_yolo(cycle_name, str(yaml_path.absolute()), dataset_name, epochs=epochs, model_path=str(baseline_model.absolute()))

            print(f"Resultados do ciclo {cycle_name}: {results}")

            model = Path(dataset_name) / cycle_name / "weights" / "best.pt"

            evaluate_yolo(
                model_path=str(model.absolute()),
                yaml_path=str(yaml_path.absolute()),
                output_dir=labeled_txt.parent,
                name=cycle_name,
                project=dataset_name
            )

        if i < num_total_cycles - 1:
        # Lê a lista de caminhos não rotulados do arquivo de texto
            with open(unlabeled_txt, 'r') as f:
                image_paths_for_prediction = [line.strip() for line in f if line.strip()]
            
            generate_lightly_predictions(
                model_path=str(model.absolute()), # Carrega o modelo treinado
                image_paths=image_paths_for_prediction,
                output_dir=LIGHTLY_INPUT / '.lightly' / 'predictions' / 'object_detection',
                batch_size=32
            )

    move_folder(dataset_name, f'runs/{dataset_name}')
    


if __name__ == "__main__":
    main()


