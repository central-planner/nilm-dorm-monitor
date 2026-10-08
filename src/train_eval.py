"""
阶段三：模型训练与评估
======================
流程：
    CSV -> 按录制段聚合特征 -> GroupShuffleSplit 划分 -> 三模型对照 -> 报告 + 图表

⚠️ 数据泄漏红线（导师 P0 点名的坑）：
    同一录制段内相邻采样点几乎一样。若逐行随机划分训练/测试集，
    测试集的"邻居"都在训练集里出现过，准确率会虚高到 99%，答辩一问就崩。
    正确做法：按 recording_id 整段划分（GroupShuffleSplit），
    同一录制段要么全在训练集、要么全在测试集。

模型选择（与开题报告 §7.3 一致）：
    - 决策树（主力）：可解释、能画树图放进报告、几乎不调参
    - kNN / SVM（对照组）：证明"不是模型选得好，是特征本身有效"
"""

import os
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)

from . import config as C
from .features import extract_features, features_to_vector, FEATURE_ORDER


def load_feature_table(data_dir=None):
    """
    读取 data/simulated/*.csv（跳过事件序列），对每段录制提取特征。
    返回 DataFrame: [recording_id, label] + FEATURE_ORDER
    """
    data_dir = data_dir or os.path.join(C.DATA_DIR, "simulated")
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*.csv"))):
        name = os.path.basename(path)
        if name.startswith("event_"):           # 事件序列留给阶段四
            continue
        df = pd.read_csv(path)
        for rid, seg in df.groupby("recording_id"):
            feats, _ = extract_features(seg["current"].to_numpy(),
                                        seg["voltage"].to_numpy())
            rows.append({"recording_id": rid, "label": seg["label"].iloc[0], **feats})
        print(f"  [feat] {name}: {df['recording_id'].nunique()} 段")
    table = pd.DataFrame(rows)
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    table.to_csv(os.path.join(C.RESULTS_DIR, "feature_table.csv"), index=False)
    return table


def get_models():
    """三个候选模型。kNN/SVM 对量纲敏感，必须套标准化管线；决策树不需要。"""
    return {
        "DecisionTree(d=5)": DecisionTreeClassifier(max_depth=5, random_state=C.RANDOM_SEED),
        "kNN(k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "SVM(rbf)": make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale",
                                                        random_state=C.RANDOM_SEED)),
    }


def run_stage3(feature_table=None, verbose=True):
    """阶段三主流程。返回 (主模型, 评估结果 dict)。"""
    if feature_table is None:
        feature_table = load_feature_table()

    X = feature_table[FEATURE_ORDER].to_numpy()
    y = feature_table["label"].to_numpy()
    groups = feature_table["recording_id"].to_numpy()

    # ---- 按录制段分组划分（修复数据泄漏的关键一步）----
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=C.RANDOM_SEED)
    tr_idx, te_idx = next(splitter.split(X, y, groups))
    Xtr, Xte, ytr, yte = X[tr_idx], X[te_idx], y[tr_idx], y[te_idx]

    # ---- 三模型对照 + 分组 5 折交叉验证 ----
    gkf = GroupKFold(n_splits=5)
    results, fitted = {}, {}
    for name, model in get_models().items():
        cv = cross_val_score(model, Xtr, ytr, groups=groups[tr_idx], cv=gkf)
        model.fit(Xtr, ytr)
        acc = accuracy_score(yte, model.predict(Xte))
        results[name] = {"cv_mean": cv.mean(), "cv_std": cv.std(), "test_acc": acc}
        fitted[name] = model
        if verbose:
            print(f"  [ml] {name:20s} CV={cv.mean():.3f}±{cv.std():.3f}  测试集={acc:.3f}")

    # ---- 主模型（决策树）的详细评估 ----
    best_name = "DecisionTree(d=5)"
    best = fitted[best_name]
    y_pred = best.predict(Xte)
    report = classification_report(yte, y_pred, digits=3)
    cm = confusion_matrix(yte, y_pred, labels=best.classes_)

    os.makedirs(C.FIG_DIR, exist_ok=True)

    # 混淆矩阵图
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=[s[:12] for s in best.classes_]).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_title("Confusion Matrix - Decision Tree (hold-out by recording)")
    plt.xticks(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "stage3_confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # 决策树结构图（报告里最能体现"可解释性"的一张图）
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(best, feature_names=FEATURE_ORDER,
              class_names=[s[:12] for s in best.classes_],
              filled=True, rounded=True, fontsize=9, ax=ax)
    ax.set_title("Decision Tree (max_depth=5)")
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "stage3_decision_tree.png"), dpi=150)
    plt.close(fig)

    # 特征散点图：THD vs PF（最能拉开 5 类设备的两个特征）
    fig, ax = plt.subplots(figsize=(8, 6))
    for label, grp in feature_table.groupby("label"):
        ax.scatter(grp["thd"], grp["pf"], s=25, alpha=0.7, label=label[:16])
    ax.set_xlabel("THD (total harmonic distortion)")
    ax.set_ylabel("PF (power factor)")
    ax.set_title("Feature space: THD vs PF")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(C.FIG_DIR, "stage3_feature_scatter.png"), dpi=150)
    plt.close(fig)

    # ---- 文字报告 ----
    os.makedirs(C.RESULTS_DIR, exist_ok=True)
    lines = ["# 阶段三评估报告（模拟数据）", "",
             f"- 样本：{len(feature_table)} 段录制（5 类设备）",
             f"- 划分：GroupShuffleSplit，训练 {len(tr_idx)} 段 / 测试 {len(te_idx)} 段（按录制段整段划分）",
             "", "## 模型对照", ""]
    for name, r in results.items():
        lines.append(f"- {name}: CV {r['cv_mean']:.3f}±{r['cv_std']:.3f} / 测试集 {r['test_acc']:.3f}")
    lines += ["", f"## 主模型 {best_name} 分类报告", "", "```", report, "```"]
    with open(os.path.join(C.RESULTS_DIR, "stage3_report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return best, {"results": results, "report": report, "feature_table": feature_table}
