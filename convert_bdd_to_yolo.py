import json
import os
from tqdm import tqdm
from PIL import Image

# --- CONFIGURAÇÕES ---

# 1. DEFINIÇÃO DA ESTRUTURA DE DIRETÓRIOS
# O script assume que está rodando no diretório pai da pasta 'bdd100k'
# BASE_DIR = 'bdd100k'
BASE_DIR = 'd10k'

PATHS = {
    # Entradas (Inputs)
    'images_train': os.path.join(BASE_DIR, 'images', 'train'),
    'images_val': os.path.join(BASE_DIR, 'images', 'val'),
    'json_train': os.path.join(BASE_DIR, 'labels', 'det_train.json'),
    'json_val': os.path.join(BASE_DIR, 'labels', 'det_val.json'),

    # Saídas (Outputs)
    'output_labels_train': os.path.join(BASE_DIR, 'labels', 'train'),
    'output_labels_val': os.path.join(BASE_DIR, 'labels', 'val'),
    'output_config_dir': BASE_DIR, # Onde salvar .yaml e .names
}

# 2. MAPEAMENTO DE CLASSES (BDD100k -> YOLO)
# Dicionário para mapear os nomes de categoria do JSON do BDD100k para o ID numérico desejado.
bdd_category_map = {
    # 'pedestrian': 0,
    'person': 0,  
    # 'bicycle': 1,        # O JSON usa 'bike'
    'bike': 1,        # O JSON usa 'bike', mas no YOLO é 'bicycle'
    'car': 2,
    # 'motorcycle': 3,       # O JSON usa 'motor'
    'motor': 3,
    'bus': 4,
    'truck': 5,
    'traffic sign': 6,
    'traffic light': 7,
    'rider': 0,  # 'rider' é tratado como 'person' no YOLO
}

# Lista com os nomes finais das classes na ordem correta dos IDs para o arquivo YAML.
yolo_class_names = [
    'person',       # 0
    'bicycle',      # 1
    'car',          # 2
    'motorcycle',   # 3
    'bus',          # 4
    'truck',       # 5
    'traffic sign', # 6
    'traffic light' # 7
]

# Categorias a serem ignoradas durante a conversão
ignore_categories = ["drivable area", "lane", "trailer", "other person", "train"] #"bike"


# --- FUNÇÕES DE CONVERSÃO (SEM ALTERAÇÕES NA LÓGICA INTERNA) ---

def convert_bdd_to_yolo(json_path, image_dir, output_label_dir):
    """
    Converte anotações do BDD100k (JSON) para o formato YOLO (.txt).
    """
    print(f"Verificando anotações em: {json_path}")
    if not os.path.exists(json_path):
        print(f"AVISO: Arquivo de anotações não encontrado: {json_path}. Pulando...")
        return

    os.makedirs(output_label_dir, exist_ok=True)
    
    with open(json_path) as f:
        data = json.load(f)

    for img_data in tqdm(data, desc=f"Convertendo labels para {os.path.basename(output_label_dir)}"):
        img_name = img_data['name']
        image_path = os.path.join(image_dir, img_name)
        
        if not os.path.exists(image_path):
            continue

        try:
            with Image.open(image_path) as img:
                img_w, img_h = img.size
        except IOError:
            print(f"Erro ao ler a imagem: {image_path}. Pulando.")
            continue

        output_txt_path = os.path.join(output_label_dir, img_name.replace('.jpg', '.txt'))
        
        with open(output_txt_path, 'w') as f_label:
            if 'labels' not in img_data:
                continue
                
            for label in img_data['labels']:
                category = label.get('category')
                if not category or category in ignore_categories or category not in bdd_category_map:
                    continue

                if 'box2d' not in label:
                    continue

                box = label['box2d']
                x1, y1, x2, y2 = box['x1'], box['y1'], box['x2'], box['y2']

                bbox_center_x = (x1 + x2) / 2
                bbox_center_y = (y1 + y2) / 2
                bbox_width = x2 - x1
                bbox_height = y2 - y1

                norm_center_x = bbox_center_x / img_w
                norm_center_y = bbox_center_y / img_h
                norm_width = bbox_width / img_w
                norm_height = bbox_height / img_h
                
                class_id = bdd_category_map[category]

                f_label.write(f"{class_id} {norm_center_x} {norm_center_y} {norm_width} {norm_height}\n")

def generate_image_list_file(image_dir, output_file_path):
    """
    Cria um arquivo .txt com o caminho absoluto de cada imagem no diretório.
    """
    print(f"Gerando lista de arquivos de imagem em: {output_file_path}")
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    with open(output_file_path, 'w') as f:
        for filename in tqdm(sorted(os.listdir(image_dir)), desc=f"Listando imagens em {os.path.basename(image_dir)}"):
            if filename.lower().endswith(".jpg"):
                full_path = os.path.abspath(os.path.join(image_dir, filename))
                f.write(full_path + "\n")

def generate_yolo_config_files(output_dir):
    """Cria os arquivos de configuração .names e .yaml para o YOLO."""
    # 1. Criar o arquivo .names
    names_path = os.path.join(output_dir, 'bdd100k.names')
    print(f"Criando arquivo de nomes em: {names_path}")
    with open(names_path, 'w') as f:
        for name in yolo_class_names:
            f.write(name + "\n")

    # 2. Criar o arquivo .yaml no novo formato
    yaml_path = os.path.join(output_dir, 'bdd100k.yaml')
    print(f"Criando arquivo de configuração YAML em: {yaml_path}")

    # Construir o bloco de nomes no formato id: nome
    names_block = "\n".join([f"  {i}: {name}" for i, name in enumerate(yolo_class_names)])

    # Conteúdo do arquivo YAML
    yaml_content = f"""
# Diretório raiz do dataset. Edite o path se o seu dataset não for um subdiretório do projeto.
path: ./{os.path.basename(output_dir)}

# Conjuntos de treino e validação (caminhos relativos ao 'path' acima)
train: images/train
val: images/val
test: images/test

# Definição das Classes
names:
{names_block}
"""
    with open(yaml_path, 'w') as f:
        f.write(yaml_content.strip())
    print("\nArquivos de configuração gerados com sucesso!")
# -----------------------------------



# --- BLOCO DE EXECUÇÃO ---

if __name__ == '__main__':
    # 1. Converter anotações JSON para formato YOLO .txt
    convert_bdd_to_yolo(PATHS['json_train'], PATHS['images_train'], PATHS['output_labels_train'])
    convert_bdd_to_yolo(PATHS['json_val'], PATHS['images_val'], PATHS['output_labels_val'])

    # 2. Gerar os arquivos de configuração .yaml e .names
    # (Não precisa mais gerar train.txt e val.txt, pois o novo yaml usa caminhos diretos)
    generate_yolo_config_files(PATHS['output_config_dir'])

    print("\nConversão do dataset BDD100k para o formato YOLO concluída! ✅")