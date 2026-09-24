"""Images PNG des plots DVC pour le rapport CML.

Usage : python src/cml_plots.py [REVISION_DE_REFERENCE]
Compare l'espace de travail (nouvelle version) à une révision Git (ex. origin/main).
"""
import json
import os
import subprocess
import sys
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "cml_plots"
BASE = sys.argv[1] if len(sys.argv) > 1 else None


def load(path, rev=None):
    try:
        if rev is None:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        txt = subprocess.run(["git", "show", f"{rev}:{path}"],
                             capture_output=True, text=True, check=True).stdout
        return json.loads(txt)
    except Exception:
        return None


def versions(path):
    out = []
    ref = load(path, BASE) if BASE else None
    if ref:
        out.append(("référence", ref))
    cur = load(path)
    if cur:
        out.append(("nouvelle version", cur))
    return out


os.makedirs(OUT, exist_ok=True)

# 1) Matrice de confusion : une matrice par version
data = versions("plots/confusion_matrix.json")
if data:
    classes = sorted({r["actual"] for _, d in data for r in d}
                     | {r["predicted"] for _, d in data for r in d})
    fig, axes = plt.subplots(1, len(data), figsize=(4.5 * len(data), 4), squeeze=False)
    for ax, (label, d) in zip(axes[0], data):
        counts = Counter((r["actual"], r["predicted"]) for r in d)
        m = [[counts[(a, p)] for p in classes] for a in classes]
        ax.imshow(m, cmap="Blues")
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, m[i][j], ha="center", va="center")
        ax.set_xticks(range(len(classes)), classes, rotation=30)
        ax.set_yticks(range(len(classes)), classes)
        ax.set_xlabel("prédit")
        ax.set_ylabel("réel")
        ax.set_title(label)
    fig.suptitle("Matrice de confusion")
    fig.tight_layout()
    fig.savefig(f"{OUT}/confusion_matrix.png", dpi=110)
    plt.close(fig)

# 2) Importance des variables : barres groupées
data = versions("plots/feature_importance.json")
if data:
    feats = [r["feature"] for r in data[-1][1]]
    h = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(6.5, 3.5))
    for k, (label, d) in enumerate(data):
        imp = {r["feature"]: r["importance"] for r in d}
        ax.barh([i + k * h for i in range(len(feats))],
                [imp.get(f, 0) for f in feats], height=h, label=label)
    ax.set_yticks([i + h * (len(data) - 1) / 2 for i in range(len(feats))], feats)
    ax.invert_yaxis()
    ax.set_xlabel("importance")
    ax.set_title("Importance des variables")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/feature_importance.png", dpi=110)
    plt.close(fig)

# 3) Courbe ROC : courbes superposées
data = versions("plots/roc_curve.json")
if data:
    fig, ax = plt.subplots(figsize=(5, 4))
    for label, d in data:
        ax.plot([r["fpr"] for r in d], [r["tpr"] for r in d], label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("fpr")
    ax.set_ylabel("tpr")
    ax.set_title("Courbe ROC (micro-moyenne)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/roc_curve.png", dpi=110)
    plt.close(fig)

print("[OK] images générées dans", OUT)