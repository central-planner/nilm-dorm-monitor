# 阶段三评估报告（模拟数据）

- 样本：100 段录制（5 类设备）
- 划分：GroupShuffleSplit，训练 70 段 / 测试 30 段（按录制段整段划分）

## 模型对照

- DecisionTree(d=5): CV 1.000±0.000 / 测试集 1.000
- kNN(k=5): CV 0.986±0.029 / 测试集 1.000
- SVM(rbf): CV 1.000±0.000 / 测试集 1.000

## 主模型 DecisionTree(d=5) 分类报告

```
                  precision    recall  f1-score   support

 E1_incandescent      1.000     1.000     1.000         7
E2_led_desk_lamp      1.000     1.000     1.000         6
  E3_usb_charger      1.000     1.000     1.000         6
       E4_ac_fan      1.000     1.000     1.000         6
  E5_dimmer_lamp      1.000     1.000     1.000         5

        accuracy                          1.000        30
       macro avg      1.000     1.000     1.000        30
    weighted avg      1.000     1.000     1.000        30

```