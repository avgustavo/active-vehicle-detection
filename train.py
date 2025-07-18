import argparse
from pathlib import Path
from lightly.api import ApiWorkflowClient
from ultralytics import YOLO
import os
import comet_ml
import shutil

######################################## CONSTANTES ########################################
LIGHTLY_TOKEN = "6ef4b5e20f6a1dba87a72a9eb4ddceb3f9529cd3d46b94a8" 
DATASET_PATH = 'FOCAL/yolov5_format/images'


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
    client = ApiWorkflowClient(token=LIGHTLY_TOKEN)
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
    tag_files = []
    image_names = ''
    for tag in tags:
        print(tag.name)
        if image_names == '':
            image_names = client.export_filenames_by_tag_name(tag_name=tag.name)
        else:
            # Adiciona uma nova linha antes de adicionar os nomes da próxima tag
            image_names += '\n' + client.export_filenames_by_tag_name(tag_name=tag.name)

        file_name = f"{tag.name}.txt"
        file_path = DATASET_PATH + "/" + file_name
        
        with open(file_path, "w") as f:
            f.write(image_names)
        tag_files.append(file_name)

    print(f"Arquivos de texto criados! =D")

    return tag_files


def prepare_yolo_dataset(cycle_name: str, file: str, dataset_path: str, dataset_name: str) -> Path:
    
    output_dir = Path(f"runs/{dataset_name}/config/{cycle_name}")
    os.makedirs(output_dir, exist_ok=True)    
    # Criando o data.yaml
    yaml = f"""
path: {dataset_path}
train: {file}
val: val 
test: test
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

    return str(output_dir / "data.yaml")

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
    parse.add_argument("--dataset_name", type=str, required=True, help="Name of the dataset")

    args = parse.parse_args()

    dataset_name = args.dataset_name

    comet_ml.login(project_name=dataset_name)

    create_dir(dataset_name)
    
    tag_files = lightly_init(dataset_name)

    for file in tag_files:
        cycle_name = file.split('.')[0]
        file_path = Path(DATASET_PATH) / file


        # Preparar o dataset para o YOLO
        yaml_path = prepare_yolo_dataset(cycle_name, file, DATASET_PATH, dataset_name)

        print(yaml_path.split('data.yaml')[0] + file)
        
        shutil.copyfile(file_path, yaml_path.split('data.yaml')[0] + file)

        # Treinar o modelo YOLO
        train_yolo(cycle_name, yaml_path, dataset_name, epochs=0)
    
    


if __name__ == "__main__":
    main()