"""
NILM 项目一键管线
=================
用法：
    python run_pipeline.py            # 完整跑：模拟数据 -> 阶段三 -> 阶段四
    python run_pipeline.py --skip-sim # 已有数据，直接跑阶段三+四

真实数据替换方法：
    把采集到的 CSV 放进 data/real/（列格式与模拟数据一致：
    timestamp, voltage, current, label, recording_id），
    然后用 --data-dir data/real 运行即可，其余代码零改动。
"""

import argparse
import os

from src import config as C
from src.simulator import generate_dataset, generate_event_sequence
from src.train_eval import load_feature_table, run_stage3
from src.event_detect import run_stage4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sim", action="store_true", help="跳过模拟数据生成")
    ap.add_argument("--data-dir", default=None, help="单设备数据目录（默认 data/simulated）")
    args = ap.parse_args()

    data_dir = args.data_dir or os.path.join(C.DATA_DIR, "simulated")

    print("=" * 60)
    print("[1/3] 数据准备")
    print("=" * 60)
    if not args.skip_sim:
        generate_dataset(out_dir=data_dir)
        generate_event_sequence()
    else:
        print(f"  使用已有数据: {data_dir}")

    print("\n" + "=" * 60)
    print("[2/3] 阶段三：特征提取 + 模型训练评估")
    print("=" * 60)
    table = load_feature_table(data_dir)
    _, stage3 = run_stage3(table)
    acc3 = stage3["results"]["DecisionTree(d=5)"]["test_acc"]
    print(f"\n  阶段三结论：决策树测试集准确率 = {acc3:.1%}（目标 ≥90%）")

    print("\n" + "=" * 60)
    print("[3/3] 阶段四：事件检测 + 差分匹配")
    print("=" * 60)
    _, acc4 = run_stage4(table)

    print("\n" + "=" * 60)
    print("管线完成。产出：")
    print(f"  图表   -> {C.FIG_DIR}")
    print(f"  报告   -> {C.RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
