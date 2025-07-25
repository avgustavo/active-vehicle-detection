import os
import shutil
import random

def resize_dataset(
    val_images_dir: str,
    train_images_dir: str,
    val_labels_dir: str,
    train_labels_dir: str,
    num_to_move: int,
    label_extension: str = '.txt'
):
    """
    Move um número específico de imagens e suas labels correspondentes
    do conjunto de validação para o conjunto de treino.

    Args:
        val_images_dir (str): Caminho para a pasta de imagens de validação.
        train_images_dir (str): Caminho para a pasta de imagens de treino.
        val_labels_dir (str): Caminho para a pasta de labels de validação.
        train_labels_dir (str): Caminho para a pasta de labels de treino.
        num_to_move (int): O número de arquivos a serem movidos.
        label_extension (str): A extensão do arquivo de label (ex: '.txt', '.xml', '.json').
    """
    print("--- Iniciando o Rebalanceamento dos Datasets ---")
    
    # 1. Garantir que os diretórios de destino existam
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    
    # 2. Listar todas as imagens disponíveis na pasta de validação
    try:
        available_images = [f for f in os.listdir(val_images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Encontradas {len(available_images)} imagens no diretório de validação.")
    except FileNotFoundError:
        print(f"ERRO: O diretório de imagens de validação não foi encontrado em: {val_images_dir}")
        return

    # 3. Verificar se temos imagens suficientes para mover
    if len(available_images) < num_to_move:
        print(f"ERRO: Você quer mover {num_to_move} imagens, mas apenas {len(available_images)} estão disponíveis.")
        return

    # 4. Selecionar aleatoriamente as imagens que serão movidas
    # Usamos random.sample para garantir que não haja repetições
    random.seed(42)  # Para reprodutibilidade
    images_to_move = random.sample(available_images, num_to_move)
    print(f"Selecionando aleatoriamente {len(images_to_move)} imagens para mover...")

    # 5. Mover cada imagem e sua label correspondente
    moved_count = 0
    warning_count = 0
    for image_name in images_to_move:
        # Derivar o nome do arquivo de label a partir do nome da imagem
        base_name, _ = os.path.splitext(image_name)
        label_name = base_name + label_extension

        # Definir os caminhos de origem e destino
        src_image_path = os.path.join(val_images_dir, image_name)
        dst_image_path = os.path.join(train_images_dir, image_name)
        
        src_label_path = os.path.join(val_labels_dir, label_name)
        dst_label_path = os.path.join(train_labels_dir, label_name)

        # Mover o arquivo de imagem
        try:
            shutil.move(src_image_path, dst_image_path)
        except FileNotFoundError:
            print(f"AVISO: Arquivo de imagem não encontrado em {src_image_path}. Pulando.")
            continue
            
        # Mover o arquivo de label (com verificação de existência)
        if os.path.exists(src_label_path):
            shutil.move(src_label_path, dst_label_path)
            moved_count += 1
        else:
            print(f"AVISO: Label correspondente '{label_name}' não encontrada para a imagem '{image_name}'. A imagem foi movida, mas a label não.")
            warning_count += 1

    print("\n--- Rebalanceamento Concluído ---")
    print(f"Total de pares (imagem + label) movidos com sucesso: {moved_count}")
    if warning_count > 0:
        print(f"Avisos de labels não encontradas: {warning_count}")
    print("Verifique as contagens de arquivos nas pastas para confirmar.")


if __name__ == '__main__':
    # --- CONFIGURE AQUI ---
    
    # Defina a estrutura de pastas do seu projeto
    # Exemplo para um formato comum como o do YOLO
    base_dir = 'treino_transitar'
    
    VAL_IMAGES_PATH = os.path.join(base_dir, 'val/images')
    TRAIN_IMAGES_PATH = os.path.join(base_dir, 'train/images')
    
    VAL_LABELS_PATH = os.path.join(base_dir, 'val/labels')
    TRAIN_LABELS_PATH = os.path.join(base_dir, 'train/labels')
    
    # Número de imagens a mover (conforme calculado)
    NUM_IMAGES_TO_MOVE = 1118
    
    # Extensão dos seus arquivos de anotação
    LABEL_FILE_EXTENSION = '.txt'  # Mude para '.xml' ou '.json' se necessário

    # --- FIM DA CONFIGURAÇÃO ---
    
    # DICA DE PROFISSIONAL: Faça um backup antes de rodar scripts que modificam arquivos!
    print("AVISO: Este script irá mover arquivos permanentemente.")
    # Descomente a linha abaixo para rodar o script após confirmar os caminhos
    # input("Pressione Enter para continuar ou Ctrl+C para cancelar...")

    # Chama a função principal
    resize_dataset(
        val_images_dir=VAL_IMAGES_PATH,
        train_images_dir=TRAIN_IMAGES_PATH,
        val_labels_dir=VAL_LABELS_PATH,
        train_labels_dir=TRAIN_LABELS_PATH,
        num_to_move=NUM_IMAGES_TO_MOVE,
        label_extension=LABEL_FILE_EXTENSION
    )
    
    # Verificação final (opcional, mas recomendado)
    print("\n--- Verificação Final ---")
    try:
        final_train_count = len(os.listdir(TRAIN_IMAGES_PATH))
        final_val_count = len(os.listdir(VAL_IMAGES_PATH))
        print(f"Novas contagens -> Treino: {final_train_count} imagens | Validação: {final_val_count} imagens.")
        
        total = final_train_count + final_val_count
        print(f"Proporção aproximada -> Treino: {final_train_count/total:.1%} | Validação: {final_val_count/total:.1%}")
    except FileNotFoundError:
        print("Não foi possível verificar as contagens finais. Verifique os caminhos.")