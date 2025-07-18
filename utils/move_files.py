import os
import shutil

def move_files(origem, destino):
    """
    Move todos os arquivos de uma pasta de origem para uma pasta de destino.

    Args:
        origem (str): O caminho da pasta de origem.
        destino (str): O caminho da pasta de destino.
    """
    if not os.path.exists(origem):
        print(f"Erro: A pasta de origem '{origem}' não existe.")
        return

    if not os.path.exists(destino):
        os.makedirs(destino) # Cria a pasta de destino se ela não existir
        print(f"Pasta de destino '{destino}' criada.")

    for nome_arquivo in os.listdir(origem):
        caminho_origem = os.path.join(origem, nome_arquivo)
        caminho_destino = os.path.join(destino, nome_arquivo)

        # Verifica se é um arquivo (e não uma subpasta)
        if os.path.isfile(caminho_origem):
            try:
                shutil.move(caminho_origem, caminho_destino)
                print(f"Arquivo '{nome_arquivo}' movido para '{destino}'.")
            except Exception as e:
                print(f"Erro ao mover o arquivo '{nome_arquivo}': {e}")
        else:
            print(f"'{nome_arquivo}' é uma pasta, será ignorada.")

def move_folder(origem, destino):
    """
    Move uma pasta inteira de origem para destino.

    Args:
        origem (str): O caminho da pasta de origem.
        destino (str): O caminho da pasta de destino.
    """
    if not os.path.exists(origem):
        print(f"Erro: A pasta de origem '{origem}' não existe.")
        return

    if not os.path.exists(destino):
        os.makedirs(destino) # Cria a pasta de destino se ela não existir
        print(f"Pasta de destino '{destino}' criada.")

    try:
        shutil.move(origem, destino)
        print(f"Pasta '{origem}' movida para '{destino}'.")
    except Exception as e:
        print(f"Erro ao mover a pasta '{origem}': {e}")