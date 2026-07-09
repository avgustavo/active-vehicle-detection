import os
import re
import warnings
import argparse
from pathlib import Path
from typing import List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import MultipleLocator

"""
Script: plot_active_learning_v3_final.py
-----------------------------------------
Gera um painel completo de visualizações para análise de experimentos de Active Learning.

Versão 3.0 (01/08/2025):
- Geração de 4 tipos de gráficos:
  1. Média Geral (Agregado)
  2. Visão Combinada (Tudo em um)
  3. Análise por Estratégia (Performance das classes dentro de uma estratégia)
  4. Análise por Classe (Comparativo das estratégias para uma classe)
- Organização dos gráficos em subpastas para maior clareza.
"""

# --- Constantes e Funções de Base (sem alterações) ---
METRIC_COLS = ["Box-P", "Box-R", "Box-F1", "mAP50", "mAP50-95"]
CSV_BASENAMES = {"res_val.csv", "results.csv"}

def scan_csv_files(root_dir: Path) -> List[Path]:
    files = []
    for name in CSV_BASENAMES:
        files.extend(root_dir.rglob(name))
    if not files:
        warnings.warn(f"Nenhum arquivo com os nomes {CSV_BASENAMES} foi encontrado em '{root_dir}'.")
    return sorted(files)

def extract_strategy_cycle(csv_path: Path):
    parts = csv_path.parts
    if len(parts) < 3:
        warnings.warn(f"Estrutura de pasta inesperada para {csv_path}. Usando 'default' como estratégia.")
        return "default", 0
    strategy = parts[-3]
    m = re.search(r"(?:ciclo|cycle|round|iter|epoch)?[_\-]?(\d+)", parts[-2], re.I)
    cycle = int(m.group(1)) if m else 0
    return strategy, cycle

def sanitize_name(name: str) -> str:
    return name.lower().replace(" ", "-").replace("/", "_")

def collect_per_class_data(csv_files: List[Path]) -> pd.DataFrame:
    all_dfs = []
    for f in csv_files:
        try:
            df = pd.read_csv(f)
            df.columns = df.columns.str.strip()
            df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
            class_col_name = next((c for c in df.columns if c.strip().lower() == "class"), None)
            if not class_col_name:
                raise ValueError("Coluna 'Class' ausente")
            df = df.rename(columns={class_col_name: "Class"})
            strategy, cycle = extract_strategy_cycle(f)
            df["strategy"] = strategy
            df["cycle"] = cycle
            for col in METRIC_COLS:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    df[col] = pd.NA
            all_dfs.append(df)
        except Exception as e:
            warnings.warn(f"[SKIP] Não foi possível processar o arquivo {f}: {e}")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


# --- Funções de Plotagem (4 tipos) ---

def plot_aggregate_metrics(df_agg: pd.DataFrame, out_dir: Path, name_prefix: str, **kwargs):
    print("Gerando gráficos da Média Geral...")
    df_filtered = df_agg[df_agg["cycle"].between(kwargs['min_cycle'], kwargs['max_cycle'])]
    for metric in METRIC_COLS:
        if df_filtered[metric].isnull().all(): continue
        plt.figure(figsize=(10, 6)); sns.set_theme(style="whitegrid")
        for strategy, grp in df_filtered.groupby("strategy"):
            grp_sorted = grp.sort_values("cycle")
            plt.plot(grp_sorted["cycle"], grp_sorted[metric], marker="o", linestyle="-", label=strategy)
        plt.xlabel("Ciclo de Active Learning"); plt.ylabel(metric)
        plt.title(f"Evolução da Média Geral de {metric}")
        ax = plt.gca(); ax.xaxis.set_major_locator(MultipleLocator(base=kwargs['x_step']))
        plt.legend(title="Estratégia"); plt.tight_layout()
        fname = f"{name_prefix}1-media-geral-{sanitize_name(metric)}.png"
        plt.savefig(out_dir / fname, dpi=300); plt.close()

def plot_combined_view(df_all: pd.DataFrame, out_dir: Path, name_prefix: str, **kwargs):
    print("Gerando gráficos da Visão Combinada...")
    df_filtered = df_all[df_all["cycle"].between(kwargs['min_cycle'], kwargs['max_cycle'])].copy()
    df_filtered["Class"] = df_filtered["Class"].astype(str).str.strip()
    for metric in METRIC_COLS:
        if df_filtered[metric].isnull().all(): continue
        plt.figure(figsize=(12, 7)); sns.set_theme(style="whitegrid")
        sns.lineplot(data=df_filtered, x="cycle", y=metric, hue="strategy", style="Class", marker="o", errorbar=None)
        plt.xlabel("Ciclo de Active Learning"); plt.ylabel(metric)
        plt.title(f"Visão Combinada: Evolução de {metric}")
        ax = plt.gca(); ax.xaxis.set_major_locator(MultipleLocator(base=kwargs['x_step']))
        plt.legend(title="Estratégia / Classe", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        fname = f"{name_prefix}2-visao-combinada-{sanitize_name(metric)}.png"
        plt.savefig(out_dir / fname, dpi=300); plt.close()

def plot_per_strategy(df_all: pd.DataFrame, out_dir: Path, name_prefix: str, **kwargs):
    print("Gerando gráficos por Estratégia...")
    df_filtered = df_all[df_all["cycle"].between(kwargs['min_cycle'], kwargs['max_cycle'])].copy()
    strategies = df_filtered['strategy'].unique()
    for strategy_name in strategies:
        df_strategy = df_filtered[df_filtered['strategy'] == strategy_name]
        for metric in METRIC_COLS:
            if df_strategy[metric].isnull().all(): continue
            plt.figure(figsize=(10, 6)); sns.set_theme(style="whitegrid")
            sns.lineplot(data=df_strategy, x="cycle", y=metric, hue="Class", style="Class", marker="o", errorbar=None)
            plt.xlabel("Ciclo de Active Learning"); plt.ylabel(metric)
            plt.title(f"Estratégia '{strategy_name}': Evolução de {metric}")
            ax = plt.gca(); ax.xaxis.set_major_locator(MultipleLocator(base=kwargs['x_step']))
            plt.legend(title="Classe"); plt.tight_layout()
            fname = f"{name_prefix}estrategia-{sanitize_name(strategy_name)}-{sanitize_name(metric)}.png"
            plt.savefig(out_dir / fname, dpi=300); plt.close()

def plot_per_class(df_all: pd.DataFrame, out_dir: Path, name_prefix: str, **kwargs):
    print("Gerando gráficos por Classe...")
    df_filtered = df_all[df_all["cycle"].between(kwargs['min_cycle'], kwargs['max_cycle'])].copy()
    classes = df_filtered['Class'].unique()
    for class_name in classes:
        df_class = df_filtered[df_filtered['Class'] == class_name]
        for metric in METRIC_COLS:
            if df_class[metric].isnull().all(): continue
            plt.figure(figsize=(10, 6)); sns.set_theme(style="whitegrid")
            sns.lineplot(data=df_class, x="cycle", y=metric, hue="strategy", style="strategy", marker="o", errorbar=None)
            plt.xlabel("Ciclo de Active Learning"); plt.ylabel(metric)
            plt.title(f"Classe '{class_name}': Comparativo de Estratégias para {metric}")
            ax = plt.gca(); ax.xaxis.set_major_locator(MultipleLocator(base=kwargs['x_step']))
            plt.legend(title="Estratégia"); plt.tight_layout()
            fname = f"{name_prefix}classe-{sanitize_name(class_name)}-{sanitize_name(metric)}.png"
            plt.savefig(out_dir / fname, dpi=300); plt.close()

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gera um painel completo de gráficos para análise de Active Learning.")
    parser.add_argument("--root_dir", default="runs", help="Diretório raiz com subpastas de estratégias")
    parser.add_argument("--out_dir", default="plots_completos", help="Pasta de saída para todos os PNGs")
    parser.add_argument("--name", "-n", default="", help="Prefixo para diferenciar arquivos gerados (ex: 'exp1-')")
    parser.add_argument("--min_cycle", type=int, default=0, help="Ciclo mínimo a considerar")
    parser.add_argument("--max_cycle", type=int, default=100, help="Ciclo máximo a considerar")
    parser.add_argument("--x_step", type=int, default=1, help="Intervalo entre ticks do eixo-x")
    args = parser.parse_args()

    if args.x_step < 1: raise ValueError("--x_step deve ser inteiro positivo")
    if args.name and not args.name.endswith('-'): args.name += '-'

    root_path = Path(args.root_dir)
    csv_files = scan_csv_files(root_path)
    if not csv_files: raise FileNotFoundError(f"Operação cancelada: Nenhum CSV encontrado em '{args.root_dir}'")

    df_per_class = collect_per_class_data(csv_files)
    if df_per_class.empty: raise RuntimeError("Nenhum dado válido foi coletado.")
    
    # Prepara os dados de média geral
    df_aggregate = df_per_class.groupby(["strategy", "cycle"])[METRIC_COLS].mean().reset_index()

    # Cria diretórios de saída
    out_path = Path(args.out_dir)
    out_path_strategy = out_path / "por_estrategia"
    out_path_class = out_path / "por_classe"
    out_path.mkdir(parents=True, exist_ok=True)
    out_path_strategy.mkdir(exist_ok=True)
    out_path_class.mkdir(exist_ok=True)

    # Dicionário de argumentos para as funções de plotagem
    plot_args = {
        "min_cycle": args.min_cycle, "max_cycle": args.max_cycle, "x_step": args.x_step
    }

    # Gera todos os conjuntos de gráficos
    plot_aggregate_metrics(df_aggregate, out_path, args.name, **plot_args)
    plot_combined_view(df_per_class, out_path, args.name, **plot_args)
    plot_per_strategy(df_per_class, out_path_strategy, args.name, **plot_args)
    plot_per_class(df_per_class, out_path_class, args.name, **plot_args)

    print(f"\n✅ Operação concluída com sucesso!")
    print(f"  - Gráficos gerais salvos em: '{out_path.resolve()}'")
    print(f"  - Gráficos por estratégia salvos em: '{out_path_strategy.resolve()}'")
    print(f"  - Gráficos por classe salvos em: '{out_path_class.resolve()}'")