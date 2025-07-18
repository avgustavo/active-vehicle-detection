from pathlib import Path
import os


# Classes que queremos detectar
CLASS_MAP = {
    'car': 0,
    'motorcycle': 1
}

# Dimensões das imagens
IMG_WIDTH = 1920
IMG_HEIGHT = 1080


def convert_bbox(bbox):
    """Converte bbox (x, y, w, h) para formato YOLO normalizado."""
    x, y, w, h = bbox
    x_center = (x + w / 2) / IMG_WIDTH
    y_center = (y + h / 2) / IMG_HEIGHT
    w /= IMG_WIDTH
    h /= IMG_HEIGHT
    return x_center, y_center, w, h


def process_annotation(txt_file, output_txt):
    with open(txt_file, 'r') as f:
        lines = f.readlines()

    label = None

    for line in lines:
        if line.startswith('position_vehicle:'):
            bbox = [int(x) for x in line.split(':')[1].strip().split()]
            bbox = convert_bbox(bbox)

        if line.startswith('type:'):
            type_vehicle = line.split(':')[1].strip()

            if type_vehicle in CLASS_MAP:
                label = f"{CLASS_MAP[type_vehicle]} {' '.join(map(str, bbox))}\n"

    if label:
        with open(output_txt, 'w') as f:
            f.write(label)


def convert_dataset(input_folder, output_folder):
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    for img_file in input_folder.rglob("*.png"):
        txt_file = img_file.with_suffix('.txt')
        if not txt_file.exists():
            continue

        img_output = output_folder / img_file.name
        txt_output = output_folder / (img_file.stem + '.txt')

        # Copia imagem
        os.system(f'cp \"{img_file}\" \"{img_output}\"')

        # Processa anotação
        process_annotation(txt_file, txt_output)

    print(f"Conversão concluída para {output_folder}")


if __name__ == "__main__":
    convert_dataset('UFPR-ALPR/training', 'data/yolo_format/train')
    convert_dataset('UFPR-ALPR/validation', 'data/yolo_format/val')
    convert_dataset('UFPR-ALPR/testing', 'data/yolo_format/test')
