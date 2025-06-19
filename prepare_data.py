import os
import re
import shutil
from pathlib import Path
from tqdm import tqdm
import yaml

# --- CONFIGURAÇÕES ---

# 1. Caminho para a pasta raiz do dataset original
SOURCE_DATA_ROOT = Path("./UFPR-ALPR")

# 2. Nome da pasta onde os dados processados serão salvos
PROCESSED_DATA_DIR = Path("./yolo_dataset")

# 3. Dimensões das imagens (conforme a descrição do dataset)
IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1080

# 4. Mapeamento de classes
CLASS_MAP = {
    'car': 0,
    'motorcycle': 1
}


def parse_object_info(annotation_path: Path) -> dict | None:
    """
    Lê um arquivo de anotação e extrai a classe e a bounding box do objeto.
    Procura por 'position_vehicle' para a bbox e 'type' para a classe.
    
    Returns:
        Um dicionário {'class_name': str, 'bbox': list[int]} ou None.
    """
    try:
        with open(annotation_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Bounding box do veículo
        bbox_match = re.search(r'position_vehicle:\s*(\d+)\s*(\d+)\s*(\d+)\s*(\d+)', content)
        # Tipo do veículo
        type_match = re.search(r'type:\s*(car|motorcycle)', content)

        if bbox_match and type_match:
            bbox = [int(coord) for coord in bbox_match.groups()]
            class_name = type_match.group(1)
            
            # Garante que a classe encontrada está no nosso mapeamento
            if class_name in CLASS_MAP:
                return {'class_name': class_name, 'bbox': bbox}

    except Exception as e:
        print(f"Erro ao ler {annotation_path}: {e}")
    
    return None

def convert_to_yolo_format(bbox: list[int], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """
    Converte a bounding box [x, y, w, h] para o formato normalizado do YOLO.
    """
    x, y, w, h = bbox
    
    x_center = (x + w / 2) / img_w
    y_center = (y + h / 2) / img_h
    norm_w = w / img_w
    norm_h = h / img_h
    
    return x_center, y_center, norm_w, norm_h

def process_split(source_split_path: Path, target_split_path: Path):
    """
    Processa um split do dataset (training, validation, ou testing),
    convertendo anotações e organizando os arquivos.
    """
    target_images_path = target_split_path / "images"
    target_labels_path = target_split_path / "labels"
    
    target_images_path.mkdir(parents=True, exist_ok=True)
    target_labels_path.mkdir(parents=True, exist_ok=True)
    
    print(f"\nProcessando split: {source_split_path.name}")
    
    source_image_paths = list(source_split_path.rglob("*.png"))
    
    for img_path in tqdm(source_image_paths, desc=f"Convertendo {source_split_path.name}"):
        annot_path = img_path.with_suffix(".txt")
        
        if not annot_path.exists():
            continue
            
        # 1. Parsear a BBox e a CLASSE do formato original
        object_info = parse_object_info(annot_path)
        
        if object_info is None:
            continue
            
        # Extrair informações do dicionário retornado
        class_name = object_info['class_name']
        bbox = object_info['bbox']
        class_id = CLASS_MAP[class_name]
        
        # 2. Converter a BBox para o formato YOLO
        yolo_bbox = convert_to_yolo_format(bbox, IMAGE_WIDTH, IMAGE_HEIGHT)
        
        # 3. Definir o caminho de destino para a imagem e o label
        target_filename_base = f"{img_path.parts[-2]}_{img_path.stem}"
        target_img_path = target_images_path / f"{target_filename_base}.png"
        target_label_path = target_labels_path / f"{target_filename_base}.txt"
        
        # 4. Escrever o arquivo de label no formato YOLO com o ID da classe correto
        with open(target_label_path, 'w', encoding='utf-8') as f:
            x_c, y_c, w, h = yolo_bbox
            f.write(f"{class_id} {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
            
        # 5. Copiar a imagem para a pasta de destino
        shutil.copy(img_path, target_img_path)

# --- SCRIPT PRINCIPAL ---

def main():
    """
    Função principal que orquestra a preparação dos dados.
    """
    if PROCESSED_DATA_DIR.exists():
        print(f"Limpando diretório de destino existente: {PROCESSED_DATA_DIR}")
        shutil.rmtree(PROCESSED_DATA_DIR)
    
    print("Iniciando a preparação dos dados para o formato YOLO (car/motorcycle)...")
    
    process_split(SOURCE_DATA_ROOT / "training", PROCESSED_DATA_DIR / "train")
    process_split(SOURCE_DATA_ROOT / "validation", PROCESSED_DATA_DIR / "valid")
    process_split(SOURCE_DATA_ROOT / "testing", PROCESSED_DATA_DIR / "test")
    
    print("\nCriando o arquivo 'dataset.yaml'...")
    
    yaml_content = {
        'path': f"../{PROCESSED_DATA_DIR.name}",
        'train': 'train/images',
        'val': 'valid/images',
        'test': 'test/images',
        'nc': len(CLASS_MAP),
        'names': list(CLASS_MAP.keys())
    }
    
    yaml_path = PROCESSED_DATA_DIR / "dataset.yaml"
    
    with open(yaml_path, 'w') as f:
        yaml.dump(yaml_content, f, sort_keys=False, default_flow_style=False)
        
    print("\n" + "="*50)
    print("✅ PREPARAÇÃO CONCLUÍDA!")
    print(f"Dados formatados para YOLO salvos em: '{PROCESSED_DATA_DIR}'")
    print(f"Arquivo de configuração gerado em: '{yaml_path}'")
    print(f"Classes: {CLASS_MAP}")
    print("="*50)
    print("\nPróximo passo: Usar este diretório e o arquivo .yaml para o script de treinamento do YOLO.")

if __name__ == "__main__":
    main()