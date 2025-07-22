import argparse
from pathlib import Path
import sys

def analisar_labels(path_labels: Path):
    """
    Analisa todos os arquivos .txt em um diretório para encontrar os índices de classe únicos.

    Args:
        path_labels (Path): O caminho para o diretório que contém os arquivos de label.
    """
    if not path_labels.is_dir():
        print(f"Erro: O caminho '{path_labels}' não é um diretório válido.", file=sys.stderr)
        return

    print(f"Analisando arquivos de label no diretório: '{path_labels}'...")
    
    unique_classes = set()
    label_files = list(path_labels.glob('**/*.txt')) # Procura em subdiretórios também

    if not label_files:
        print("Nenhum arquivo .txt de label encontrado.", file=sys.stderr)
        return
    one_label = [0,0,0,0]
    # i = 0
    # count = 0
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_index = int(parts[0])
                            unique_classes.add(class_index)
                            # if class_index == i:
                            one_label[class_index] += 1
                            # i += 1
                            # image = f"{label_file.absolute()}".replace('labels', 'images')
                            # image = image.replace('.txt', '.jpg')
                            # print(f"Índice de classe {class_index} encontrado no arquivo {image}.")
                        except ValueError:
                            # Ignora linhas mal formatadas
                            pass
            # count += 1
            # if i == 4:
            #     break
        except Exception as e:
            print(f"Não foi possível ler o arquivo {label_file}: {e}", file=sys.stderr)

    if not unique_classes:
        print("Nenhuma classe encontrada nos arquivos de label.")
    else:
        sorted_classes = sorted(list(unique_classes))
        max_class = max(sorted_classes)
        num_classes = max_class + 1
        # print(f"Número total de arquivos analisados: {count}")
        print(f"\nTotal de labels encontrados: {sum(one_label)}, com a distribuição por classe: {one_label}")
        print("\n--- Resultados da Análise ---")
        print(f"Índices de classe únicos encontrados: {sorted_classes}")
        print(f"Índice máximo encontrado: {max_class}")
        print(f"Número total de classes inferido: {num_classes} (índices de 0 a {max_class})")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verifica os índices de classe em arquivos de label no formato YOLO.")
    parser.add_argument("--pth", type=str, required=True, help="Caminho para o diretório raiz dos arquivos de label (ex: FOCAL/yolov5_format/labels/train).")
    
    args = parser.parse_args()
    
    analisar_labels(Path(args.pth))