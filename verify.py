import argparse
from pathlib import Path
import sys
import pandas as pd

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

def contar_classes_por_status(csv_name, labels_dir, status=1):
    """
    Conta a distribuição das classes para imagens com determinado status no CSV.

    Args:
        csv_name (str or Path): Caminho para o CSV com colunas 'filename' e 'status'.
        labels_dir (str or Path): Diretório raiz dos arquivos de label.
        status (int or str): Status a ser filtrado (ex: 1 ou '01').

    Returns:
        dict: Distribuição das classes {classe: contagem}
    """
    csv = Path(csv_name)
    df = pd.read_csv(csv, dtype={'status': int, 'filename': str})
    status = int(status)
    selecionados = df[df['status'] == status]['filename']

    class_counts = {}

    for fname in selecionados:
        label_path = Path(labels_dir) / fname
        label_path = label_path.with_suffix('.txt')
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_idx = int(parts[0])
                            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
                        except ValueError:
                            continue
    print(f"Distribuição das classes para status={status}: {class_counts}")
    return class_counts

def contar_classes_por_txt(txt_path, labels_dir):
    """
    Conta a distribuição das classes para imagens listadas em um arquivo de texto.

    Args:
        txt_path (str or Path): Caminho para o arquivo de texto com caminhos das imagens (um por linha).
        labels_dir (str or Path): Diretório raiz dos arquivos de label.

    Returns:
        dict: Distribuição das classes {classe: contagem}
    """
    txt_path = Path(txt_path)
    labels_dir = Path(labels_dir)
    with open(txt_path, 'r') as f:
        imagens = [line.strip() for line in f if line.strip()]

    class_counts = {}
    for img_path in imagens:
        img_stem = Path(img_path).stem
        label_file = labels_dir / f"{img_stem}.txt"
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts:
                        try:
                            class_idx = int(parts[0])
                            class_counts[class_idx] = class_counts.get(class_idx, 0) + 1
                        except ValueError:
                            continue
    print(f"Distribuição das classes para imagens do arquivo '{txt_path}': {class_counts}")
    return class_counts

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Verifica os índices de classe em arquivos de label no formato YOLO.")
    parser.add_argument("-p","--path", type=str, required=True, help="Caminho para o diretório raiz dos arquivos de label (ex: FOCAL/yolov5_format/labels/train).")
    parser.add_argument("-c","--csv", type=str, help="Caminho para o CSV com colunas 'filename' e 'status' (opcional).")
    parser.add_argument("-s","--status", type=str, default='1', help="Status a ser filtrado no CSV (padrão: '1').")
    parser.add_argument("-t","--type", type=str, default='count', choices=['count', 'analise', 'txt'], help="Tipo de função a ser executada (padrão: 'count').")
    parser.add_argument("-x","--txt", type=str, help="Caminho para o arquivo de texto com imagens selecionadas (opcional, para --type txt).")
    
    args = parser.parse_args()

    if args.type == 'count':
        if not args.csv:
            print("Erro: Para a função 'count', é necessário fornecer o caminho para o CSV.", file=sys.stderr)
            sys.exit(1)
        contar_classes_por_status(args.csv, args.path, args.status)
    elif args.type == 'analise':
        analisar_labels(Path(args.path))
    elif args.type == 'txt':
        if not args.txt:
            print("Erro: Para a função 'txt', é necessário fornecer o caminho para o arquivo de texto.", file=sys.stderr)
            sys.exit(1)
        contar_classes_por_txt(args.txt, args.path)
