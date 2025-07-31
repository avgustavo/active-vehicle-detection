import pandas as pd
from pathlib import Path

# ----- defina seus caminhos -----
csv_path   = Path("lightly/.lightly/embeddings/d10_embedding.csv")        # CSV de entrada
txt_path   = Path("d10k/train_images.txt")   # arquivo-texto com nomes permitidos
out_path   = Path("lightly/.lightly/embeddings/d10_embedding.csv")  # CSV de saída
# ---------------------------------

# 1) Carrega o CSV
df = pd.read_csv(csv_path)

# 2) Lê o arquivo-texto e cria um conjunto com os nomes autorizados
with txt_path.open("r", encoding="utf-8") as f:
    nomes_permitidos = {linha.strip() for linha in f if linha.strip()}

# 3) Filtra o DataFrame
df_filtrado = df[df["filenames"].isin(nomes_permitidos)].copy()

# 4) Salva o resultado
df_filtrado.to_csv(out_path, index=False)

print(f"{len(df_filtrado)} linhas mantidas; arquivo salvo em {out_path.resolve()}")
