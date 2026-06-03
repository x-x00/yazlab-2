import os, json, logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve, average_precision_score)
from config import CFG

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

os.makedirs(CFG.plots_dir, exist_ok=True)
os.makedirs(CFG.results_dir, exist_ok=True)


# metric helpers
def aggregate_metrics(metrics_list: list) -> dict:
    # Mean ± std across seeds / folds
    keys = metrics_list[0].keys()
    return {k: {"mean": float(np.mean([m[k] for m in metrics_list])),
                "std":  float(np.std( [m[k] for m in metrics_list]))}
            for k in keys}


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    log.info(f"Saved → {path}")


# confusion matrix
def plot_confusion_matrix(y_true, y_pred, title: str, save_path: str):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Anomaly"],
                yticklabels=["Normal", "Anomaly"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved → {save_path}")


# roc / pr curves 
def plot_roc(y_true, y_score, title: str, save_path: str):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


def plot_pr(y_true, y_score, title: str, save_path: str):
    prec, rec, _ = precision_recall_curve(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(rec, prec, lw=2, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title(title); ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# automata state diagram 
def plot_automata(automata_model, title: str, save_path: str,
                  max_states: int = 15):
    try:
        import networkx as nx
    except ImportError:
        log.warning("networkx not installed - skipping automata diagram")
        return

    states, mat = automata_model.get_transition_matrix()

    if len(states) > max_states:
        out_deg = mat.sum(axis=1)
        top_idx = np.argsort(out_deg)[-max_states:]
        states = [states[i] for i in top_idx]
        mat    = mat[np.ix_(top_idx, top_idx)]

    G = nx.DiGraph()
    G.add_nodes_from(states)
    for i, src in enumerate(states):
        for j, dst in enumerate(states):
            if mat[i, j] > 0:
                G.add_edge(src, dst, weight=mat[i, j])

    fig, ax = plt.subplots(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    edge_weights = [G[u][v]["weight"] for u, v in G.edges()]
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#4C9BE8", ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=7, ax=ax)
    nx.draw_networkx_edges(G, pos, width=[w * 3 for w in edge_weights],
                           edge_color=edge_weights, edge_cmap=plt.cm.YlOrRd,
                           arrows=True, arrowsize=15, ax=ax)
    edge_labels = {(u, v): f"{G[u][v]['weight']:.2f}" for u, v in G.edges()
                   if G[u][v]["weight"] > 0.2}
    nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=6, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved → {save_path}")


# transition probability heatmap 
def plot_transition_heatmap(automata_model, title: str, save_path: str,
                            max_states: int = 20):
    states, mat = automata_model.get_transition_matrix()
    if len(states) > max_states:
        out_deg = mat.sum(axis=1)
        top_idx = np.argsort(out_deg)[-max_states:]
        states = [states[i] for i in top_idx]
        mat    = mat[np.ix_(top_idx, top_idx)]

    fig, ax = plt.subplots(figsize=(max(6, len(states) * 0.5 + 2),
                                    max(5, len(states) * 0.4 + 2)))
    sns.heatmap(mat, xticklabels=states, yticklabels=states,
                cmap="YlOrRd", annot=(len(states) <= 12),
                fmt=".2f", ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("To State"); ax.set_ylabel("From State")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved → {save_path}")


# parameter sensitivity 
def plot_param_sensitivity(results_df: pd.DataFrame, metric: str,
                           title: str, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, param in zip(axes, ["window_size", "alphabet_size"]):
        grouped = results_df.groupby(param)[f"{metric}_mean"].mean()
        stds    = results_df.groupby(param)[f"{metric}_std"].mean()
        ax.bar(grouped.index.astype(str), grouped.values,
               yerr=stds.values, capsize=4, color="#4C9BE8")
        ax.set_xlabel(param.replace("_", " ").title())
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f"{metric.upper()} vs {param}")
    fig.suptitle(title)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved → {save_path}")


def plot_state_stats(results_df: pd.DataFrame, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, col, label in [
        (axes[0], "state_count",          "Number of States"),
        (axes[1], "transition_density",   "Transition Density"),
    ]:
        if col not in results_df.columns:
            continue
        pivot = results_df.pivot_table(values=col,
                                       index="window_size",
                                       columns="alphabet_size",
                                       aggfunc="mean")
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title(label)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)


# model comparison bar chart 

def plot_model_comparison(summary: dict, dataset_name: str, save_path: str):
    metrics = ["accuracy", "precision", "recall", "f1"]
    models  = list(summary.keys())
    x       = np.arange(len(metrics))
    width   = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.Set2(np.linspace(0, 1, len(models)))
    for i, (model, color) in enumerate(zip(models, colors)):
        means = [summary[model].get(m, {}).get("mean", 0) for m in metrics]
        stds  = [summary[model].get(m, {}).get("std",  0) for m in metrics]
        ax.bar(x + i * width, means, width, yerr=stds,
               label=model, color=color, capsize=3)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(f"Model Comparison — {dataset_name}")
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120)
    plt.close(fig)
    log.info(f"Saved → {save_path}")