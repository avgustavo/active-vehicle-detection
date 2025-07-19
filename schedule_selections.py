import argparse
from os import linesep
from pathlib import Path
import platform 
from lightly.api import ApiWorkflowClient
from lightly.openapi_generated.swagger_client import DatasetType, DatasourcePurpose
import torch
import time
from typing import List

LIGHTLY_TOKEN = "6ef4b5e20f6a1dba87a72a9eb4ddceb3f9529cd3d46b94a8" 
DATASET_PATH = Path("FOCAL/yolov5_format/images/train")


def configure_lightly_client(token: str, dataset_name: str) -> ApiWorkflowClient:
    """
    Configura o cliente Lightly com o token e o dataset.
    
    Returns:
        ApiWorkflowClient: Cliente configurado.
    """
    client = ApiWorkflowClient(token=token)
    client.create_dataset(
        dataset_name=dataset_name,
        dataset_type=DatasetType.IMAGES,
    )
    client.set_local_config(purpose=DatasourcePurpose.INPUT)
    client.set_local_config(purpose=DatasourcePurpose.LIGHTLY)
    return client


def random_selection(client: ApiWorkflowClient) -> List[str]:
    '''
    Agenda um conjunto de seleções aleatórias de imagens do dataset.

    '''
    worker_config = {
        # "shutdown_when_job_finished": True,
        "use_datapool": True,
        "datasource": {
            "process_all": True,
        },
    }

    run_ids = []
    for i in range(0,10):

        if i==9:
            worker_config["shutdown_when_job_finished"] = True
            print(f"worker_config: {worker_config}")

        scheduled_run_id = client.schedule_compute_worker_run(
            worker_config=worker_config,
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
        run_ids.append(scheduled_run_id)
        time.sleep(2)  # Espera 2 segundos entre as execuções

    return run_ids


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
        f"\t-v '{dataset_path.absolute()}':/input_mount:ro \\{linesep}"
        f"\t-v '{Path('lightly').absolute()}':/lightly_mount \\{linesep}"
        f"\t-e LIGHTLY_TOKEN={lightly_token} \\{linesep}"
        f"\tlightly/worker:latest{linesep}"
        f"\033[0m"
    )
    print(
        f"{linesep}Lightly Serve command:{linesep}"
        f"\033[7m"
        f"lightly-serve \\{linesep}"
        f"\tinput_mount='{dataset_path.absolute()}' \\{linesep}"
        f"\tlightly_mount='{Path('lightly').absolute()}'{linesep}"
        f"\033[0m"
    )


def active_learning(client: ApiWorkflowClient) -> List[str]:

    worker_config = {
        # "shutdown_when_job_finished": True,
        "use_datapool": True,
        "datasource": {
            "process_all": True,
        },
    }

    run_ids = []
    for i in range(0,9):

        if i==8:
            worker_config["shutdown_when_job_finished"] = True
            print(f"worker_config: {worker_config}")

        scheduled_run_id = client.schedule_compute_worker_run(
            worker_config=worker_config,
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
        run_ids.append(scheduled_run_id)
        time.sleep(2)
    
    return run_ids


def main():
    """
    Função principal para executar a seleção aleatória e imprimir os comandos.
    """

    parse = argparse.ArgumentParser(description="Algoritmo de seleção aleatória ou de aprendizado ativo.")
    parse.add_argument("--dn", type=str, required=True, help="Nome do dataset") 
    parse.add_argument("--al", action='store_true', help="Ativa o modo de aprendizado ativo (default: aleatório)")
    args = parse.parse_args()

    dataset_name = args.dn

    client = configure_lightly_client(LIGHTLY_TOKEN, dataset_name)

    if args.al:
        print("Executando seleção de aprendizado ativo...")
        run_ids = active_learning(client)
    else:
        print("Executando seleção aleatória...")
        run_ids = random_selection(client)

    print(f"{linesep}Seleções agendadas com sucesso!{linesep}")
    print('=' * 60)
    print(f"Run IDs: {run_ids}")
    print('=' * 60)

    print_commands(DATASET_PATH, LIGHTLY_TOKEN)    

if __name__ == "__main__":
    print('Iniciando o script...')
    main()
    print("Script executado com sucesso!")