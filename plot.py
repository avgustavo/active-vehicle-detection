from pathlib import Path
import re
import warnings
import argparse
from typing import Optional

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator  # controle fino dos ticks do eixo‑x

"""
Script: **plot_active_learning.py**
----------------------------------
Curvas de evolução de métricas (YOLOv8) para diversas estratégias de Active Learning.

Novidades v0.5
==============
* **--x_step**: define o intervalo entre ticks do eixo‑x (default=1). Ex.: `--x_step 2` → 0,2,4,…
* Mantido suporte a prefixo **--name** e filtros de ciclo.

Uso rápido
----------
$ python plot_active_learning.py \
      --root_dir runs_final \
      --out_dir plots \
      --name experimentoX \
      --min_cycle 0 --max_cycle 19 \
      --x_step 2
"""

METRIC_COLS = ["Box-P", "Box-R", "Box-F1", "mAP50", "mAP50-95"]
CLASS_TOTAL_KEYS = {"all", "__all__", "overall", "total"}
CSV_BASENAMES = {"res_val.csv"}#, "results.csv"}

# -------------------------------------------------
# Helpers
# -------------------------------------------------

def scan_csv_files(root_dir: Path):
    files = []
    for name in CSV_BASENAMES:
        files.extend(root_dir.rglob(name))
    return sorted(files)


def extract_strategy_cycle(csv_path: Path):
    parts = csv_path.parts
    if len(parts) < 3:
        return "default", 0
    strategy = parts[-3]
    m = re.search(r"(?:ciclo|cycle|round|iter|epoch)?[_\-]?(\d+)", parts[-2], re.I)
    cycle = int(m.group(1)) if m else 0
    return strategy, cycle


def read_metrics_row(csv_path: Path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    df = df.drop(columns=[c for c in df.columns if c.lower().startswith("unnamed")], errors="ignore")
    class_cols = [c for c in df.columns if c.strip().lower() == "class"]
    if not class_cols:
        raise ValueError("coluna 'Class' ausente")
    if class_cols[0] != "Class":
        df = df.rename(columns={class_cols[0]: "Class"})
    for col in METRIC_COLS:
        if col not in df.columns:
            df[col] = pd.NA
        df[col] = pd.to_numeric(df[col], errors="coerce")
    mask_total = df["Class"].astype(str).str.lower().str.strip().isin(CLASS_TOTAL_KEYS)
    row = df.loc[mask_total].iloc[0] if mask_total.any() else df[METRIC_COLS].mean(numeric_only=True)
    print(row)
    return row[METRIC_COLS]


def collect_metrics(csv_files):
    recs = []
    for f in csv_files:
        try:
            metrics = read_metrics_row(f)
            strategy, cycle = extract_strategy_cycle(f)
            recs.append({"strategy": strategy, "cycle": cycle, **metrics.to_dict()})
        except Exception as e:
            warnings.warn(f"[SKIP] {f}: {e}")
    return pd.DataFrame(recs)


def sanitize_metric_name(metric: str):
    return metric.lower().replace(" ", "-")


def plot_metrics(df: pd.DataFrame, out_dir: Path, name_prefix: str, min_cycle: int, max_cycle: int, x_step: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    df = df[df["cycle"].between(min_cycle, max_cycle)]
    for metric in METRIC_COLS:
        plt.figure()
        for strategy, grp in df.groupby("strategy"):
            grp_sorted = grp.sort_values("cycle")
            plt.plot(grp_sorted["cycle"], grp_sorted[metric], marker="o", label=strategy)
        plt.xlabel("Ciclo de Active Learning")
        plt.ylabel(metric)
        plt.title(f"Evolução de {metric} (ciclos {min_cycle}-{max_cycle})")
        plt.grid(True, linestyle=":", linewidth=0.5)
        ax = plt.gca()
        ax.xaxis.set_major_locator(MultipleLocator(x_step))  # ticks de x em passo definido
        plt.legend()
        plt.tight_layout()
        fname = f"{name_prefix}-{sanitize_metric_name(metric)}.png" if name_prefix else f"{sanitize_metric_name(metric)}.png"
        plt.savefig(out_dir / fname, dpi=300)
        plt.close()


# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plota curvas de métricas de Active Learning.")
    parser.add_argument("--root_dir", default="runs", help="Diretório raiz com subpastas de estratégias")
    parser.add_argument("--out_dir", default="plots", help="Pasta de saída para PNGs")
    parser.add_argument("--name", "-n", default="", help="Prefixo para diferenciar arquivos gerados")
    parser.add_argument("--min_cycle", type=int, default=0, help="Ciclo mínimo a considerar")
    parser.add_argument("--max_cycle", type=int, default=19, help="Ciclo máximo a considerar")
    parser.add_argument("--x_step", type=int, default=2, help="Intervalo entre ticks do eixo‑x (inteiro >=1)")
    args = parser.parse_args()

    if args.x_step < 1:
        raise ValueError("--x_step deve ser inteiro positivo")

    csv_files = scan_csv_files(Path(args.root_dir))
    if not csv_files:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {args.root_dir}")

    df_metrics = collect_metrics(csv_files)
    if df_metrics.empty:
        raise RuntimeError("Nenhum dado válido coletado. Verifique estrutura e formato dos CSVs.")

    plot_metrics(
        df_metrics,
        Path(args.out_dir),
        args.name.strip(),
        args.min_cycle,
        args.max_cycle,
        args.x_step,
    )
    print(f"✅ Gráficos salvos em: {args.out_dir}")
