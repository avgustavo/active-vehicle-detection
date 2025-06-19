import os
import shutil
from pathlib import Path
from tqdm import tqdm

# Dimensões das imagens do dataset
IMG_WIDTH = 1920
IMG_HEIGHT = 1080

# ALTERAÇÃO: Mapeamento das classes de veículos para índices numéricos
CLASS_MAP = {"car": 0, "motorcycle": 1}

def get_vehicle_info(annotation_path):
    """Lê o arquivo de anotação e extrai o bounding box e o tipo do veículo."""
    vehicle_bbox = None
    vehicle_type = None
    
    with open(annotation_path, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            # Extrai a posição do veículo
            if line.strip().startswith('position_vehicle:'):
                parts = line.split(':')
                coords = parts[1].strip().split()
                vehicle_bbox = [int(c) for c in coords]
            
            # Extrai o tipo do veículo, que geralmente está na linha seguinte
            if line.strip().startswith('type:'):
                parts = line.split(':')
                # O tipo pode ser 'car' ou 'motorcycle'
                vehicle_type = parts[1].strip()

    return vehicle_bbox, vehicle_type

def convert_to_yolo_format(bbox, class_index):
    """Converte [x_min, y_min, width, height] para o formato YOLO."""
    x_min, y_min, w, h = bbox
    
    x_center = x_min + w / 2
    y_center = y_min + h / 2
    
    x_center_norm = x_center / IMG_WIDTH
    y_center_norm = y_center / IMG_HEIGHT
    w_norm = w / IMG_WIDTH
    h_norm = h / IMG_HEIGHT
    
    return f"{class_index} {x_center_norm} {y_center_norm} {w_norm} {h_norm}"

def process_dataset_split(original_split_path, yolo_base_path, split_name):
    """Processa um split do dataset (treino, validacao, teste)."""
    
    yolo_images_path = yolo_base_path / "images" / split_name
    yolo_labels_path = yolo_base_path / "labels" / split_name
    yolo_images_path.mkdir(parents=True, exist_ok=True)
    yolo_labels_path.mkdir(parents=True, exist_ok=True)
    
    image_files = sorted(list(original_split_path.glob("**/*.png")))
    
    print(f"\nProcessando {len(image_files)} imagens do split '{split_name}'...")
    
    for img_path in tqdm(image_files):
        annot_path = img_path.with_suffix(".txt")
        
        if not annot_path.exists():
            print(f"Aviso: Anotação não encontrada para {img_path}. Pulando.")
            continue
            
        # ALTERAÇÃO: Extrai info do veículo
        bbox, vehicle_type = get_vehicle_info(annot_path)
        
        if bbox and vehicle_type and vehicle_type in CLASS_MAP:
            # ALTERAÇÃO: Obtém o índice da classe
            class_index = CLASS_MAP[vehicle_type]
            
            # ALTERAÇÃO: Passa o índice da classe para a função de conversão
            yolo_string = convert_to_yolo_format(bbox, class_index)
            
            new_img_path = yolo_images_path / img_path.name
            new_label_path = yolo_labels_path / img_path.with_suffix(".txt").name
            
            shutil.copy(img_path, new_img_path)
            with open(new_label_path, 'w') as f:
                f.write(yolo_string)
        else:
            print(f"Aviso: Informação do veículo não encontrada ou tipo inválido em {annot_path}. Pulando.")


def main():
    """Função principal para orquestrar a conversão."""
    original_dataset_path = Path("../UFPR-ALPR")
    yolo_dataset_path = Path("../dataset_yolo")
    
    if yolo_dataset_path.exists():
        print(f"Pasta de destino {yolo_dataset_path} já existe. Removendo para recriar...")
        shutil.rmtree(yolo_dataset_path)

    print("Iniciando a conversão do dataset (VEÍCULOS) para o formato YOLO...")
    print(f'{original_dataset_path / "training"} - Treinamento')
    process_dataset_split(original_dataset_path / "training", yolo_dataset_path, "train")
    process_dataset_split(original_dataset_path / "validation", yolo_dataset_path, "val")
    process_dataset_split(original_dataset_path / "testing", yolo_dataset_path, "test")
    
    print("\nConversão concluída com sucesso!")
    print(f"Dataset no formato YOLO salvo em: {yolo_dataset_path.resolve()}")

if __name__ == "__main__":
    main()