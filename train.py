import comet_ml
import argparse
from pathlib import Path
from lightly.api import ApiWorkflowClient
from ultralytics import YOLO
import os
import shutil
from utils.move_files import move_folder

######################################## CONSTANTES ########################################
LIGHTLY_TOKEN = "6ef4b5e20f6a1dba87a72a9eb4ddceb3f9529cd3d46b94a8" 
DATASET_PATH = Path('FOCAL/yolov5_format')


def create_dir(dataset_name):

    if not os.path.exists(f'runs/{dataset_name}'):
        os.makedirs(f'runs/{dataset_name}')
        print(f'Diretório do dataset {dataset_name} criado com sucesso!')
    else:
        print(f'Esse diretório já existe!')


def lightly_init(dataset_name, token=LIGHTLY_TOKEN) -> list:
    """
    Inicializa o cliente lightl, define o dataset a ser utilizado
    
    Args: 
        token (str): Token de autenticação do cliente Lightly,
        dataset_name (str): Nome do dataset
    """
    #iniciar o cliente e setar dataset
    client = ApiWorkflowClient(token=token)
    client.set_dataset_id_by_name(dataset_name=dataset_name)

    # Pegando todas as tags
    tags = client.get_all_tags()

    # Ordenando na ordem correta de ciclos e tirando o initial-tag
    tags.reverse()
    tags.pop(0)

    '''
    Para cada tag, criar o arquivo de texto acumulado com os nomes no DATASET_PATH,
    pois necessita para o arquivo de configuração do YOLO.    
    '''

    caminho_abs_imagens_treino = (DATASET_PATH / 'images' / 'train').absolute()
    tag_files = []
    image_names_list = [] # Usar uma lista em vez de uma string gigante é mais limpo
    for tag in tags:
        print(tag.name)

        novos_nomes = client.export_filenames_by_tag_name(tag_name=tag.name)
        # Para cada nome de arquivo, crie o CAMINHO ABSOLUTO completo.
        for name in novos_nomes.splitlines():
            if name: # Garante que não adicionamos linhas em branco
                image_names_list.append(os.path.join(caminho_abs_imagens_treino, name.strip()))


        file_name = f"{tag.name}.txt"
        file_path = DATASET_PATH / 'images' /  file_name
        
        with open(file_path, "w") as f:
            f.write('\n'.join(image_names_list))

        tag_files.append(str(file_path.absolute()))

    print(f"Arquivos de texto criados! =D")
    return tag_files


def prepare_yolo_dataset(cycle_name: str, file_path: str, dataset_path: Path, dataset_name: str) -> Path:
    
    output_dir = Path(f"runs/{dataset_name}/config/{cycle_name}")
    os.makedirs(output_dir, exist_ok=True)    
    # Criando o data.yaml
    yaml = f"""
path: {dataset_path.absolute()}
train: {file_path}
val: images/val
test: images/test
nc: 80
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
  4: airplane
  5: bus
  6: train
  7: truck
  8: boat
  9: traffic light
  10: fire hydrant
  11: stop sign
  12: parking meter
  13: bench
  14: bird
  15: cat
  16: dog
  17: horse
  18: sheep
  19: cow
  20: elephant
  21: bear
  22: zebra
  23: giraffe
  24: backpack
  25: umbrella
  26: handbag
  27: tie
  28: suitcase
  29: frisbee
  30: skis
  31: snowboard
  32: sports ball
  33: kite
  34: baseball bat
  35: baseball glove
  36: skateboard
  37: surfboard
  38: tennis racket
  39: bottle
  40: wine glass
  41: cup
  42: fork
  43: knife
  44: spoon
  45: bowl
  46: banana
  47: apple
  48: sandwich
  49: orange
  50: broccoli
  51: carrot
  52: hot dog
  53: pizza
  54: donut
  55: cake
  56: chair
  57: couch
  58: potted plant
  59: bed
  60: dining table
  61: toilet
  62: tv
  63: laptop
  64: mouse
  65: remote
  66: keyboard
  67: cell phone
  68: microwave
  69: oven
  70: toaster
  71: sink
  72: refrigerator
  73: book
  74: clock
  75: vase
  76: scissors
  77: teddy bear
  78: hair drier
  79: toothbrush
""".strip()

    with open(output_dir / "data.yaml", "w") as f:
        f.write(yaml)

    return output_dir

def train_yolo(cycle_name, yaml_path, dataset_name, epochs=25):
    
    #Carregar modelo
    model = YOLO('yolo11n.pt')

    #Treinar modelo
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        name=f"{cycle_name}",
        project=dataset_name,
        device=[0, 1],
        plots=True,
    )
    
    return results


def main():
    parse = argparse.ArgumentParser(description="Train YOLO model with Lightly dataset")
    parse.add_argument("--dn", type=str, required=True, help="Name of the dataset")

    args = parse.parse_args()

    dataset_name = args.dn

    comet_ml.login(project_name=dataset_name)

    create_dir(dataset_name)
    
    tag_files = lightly_init(dataset_name)

    for file_path in tag_files:
        cycle_name = Path(file_path).stem

        # Preparar o dataset para o YOLO
        output_dir = prepare_yolo_dataset(cycle_name, file_path, DATASET_PATH, dataset_name)

        print(output_dir / cycle_name)

        shutil.copyfile(file_path, str(output_dir / cycle_name))

        # Treinar o modelo YOLO
        results = train_yolo(cycle_name, str(output_dir / "data.yaml"), dataset_name)

        print(f"Resultados do ciclo {cycle_name}: {results}")


        best_model_path = Path(dataset_name) / cycle_name / "weights" / "best.pt"
        if not best_model_path.exists():
            print(f"Não foi possível encontrar os pesos do melhor modelo em {best_model_path}")
            continue # Pula para o próximo ciclo

        best_model = YOLO(best_model_path)

         # Valida no conjunto 'val'
        metrics_val = best_model.val(data=str(output_dir / "data.yaml"), split='val', name=f'{cycle_name}_val', project=f'{dataset_name}')
        
        # Valida no conjunto 'test'
        # metrics_test = best_model.val(data=str(output_dir / "data.yaml"), split='test', name=f'{cycle_name}', project=f'{dataset_name}')

        val_csv = metrics_val.to_csv()
        # test_csv = metrics_test.to_csv()

        csv_filename = output_dir / "validation_results.csv"
        with open(csv_filename, "w") as f:
            f.write(val_csv)

        # csv_filename = output_dir / "test_results.csv"
        # with open(csv_filename, "w") as f:
        #     f.write(test_csv)

    move_folder(dataset_name, f'runs/{dataset_name}')
    


if __name__ == "__main__":
    main()