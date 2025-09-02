# ADT 数据处理说明

本项目处理的数据源为 `adt.csv`，该文件包含每个数据序列的基本信息。字段及格式如下：

## 数据格式说明

`adt.csv` 为逗号分隔的表格文件，每一行对应一个数据序列。字段如下：

| 字段名            | 说明                |
| -------------- | ----------------- |
| sequence\_name | 数据序列名称或唯一标识       |
| training       | 是否用于训练（1为训练，0为测试） |
| action         | 行为标签或动作类别         |

## 示例

```csv
sequence_name,training,action
work_skeleton_seq132_M1292,0,work
meal_skeleton_seq132_M1292,0,meal
decoration_skeleton_seq133_M1292,0,decoration
work_skeleton_seq131_M1292,1,work
```

## 字段说明

* **sequence\_name**：每条数据的序列名称或唯一标识，通常对应一个数据文件或文件夹。
* **training**：标记该序列是否用于训练（1）或测试（0）。
* **action**：该序列对应的动作或行为标签。

## action 字段类别及数据统计

根据当前数据，`action` 字段包含以下类别及数量：

| action     | 训练集数量 | 测试集数量 |
| ---------- | ----- | ----- |
| work       | 11    | 4     |
| meal       | 6     | 3     |
| decoration | 7     | 3     |

## 每个序列详细统计

Sequence frame statistics (原始帧数 vs 有效帧数):

| sequence\_name                                          | action     | training | frame\_num | useful\_frame | useful\_ratio |
| ------------------------------------------------------- | ---------- | -------- | ---------- | ------------- | ------------- |
| Apartment\_release\_work\_skeleton\_seq132\_M1292       | work       | test     | 3628       | 2741          | 75.55%        |
| Apartment\_release\_work\_skeleton\_seq138\_M1292       | work       | test     | 3755       | 2731          | 72.73%        |
| Apartment\_release\_meal\_skeleton\_seq132\_M1292       | meal       | test     | 3674       | 2737          | 74.50%        |
| Apartment\_release\_decoration\_skeleton\_seq133\_M1292 | decoration | test     | 3483       | 2737          | 78.58%        |
| Apartment\_release\_decoration\_skeleton\_seq139\_M1292 | decoration | test     | 4102       | 2767          | 67.45%        |
| Apartment\_release\_decoration\_skeleton\_seq134\_M1292 | decoration | test     | 3622       | 2730          | 75.37%        |
| Apartment\_release\_work\_skeleton\_seq107\_M1292       | work       | test     | 3723       | 2773          | 74.48%        |
| Apartment\_release\_meal\_skeleton\_seq135\_M1292       | meal       | test     | 3853       | 2721          | 70.62%        |
| Apartment\_release\_work\_skeleton\_seq135\_M1292       | work       | test     | 3610       | 2746          | 76.07%        |
| Apartment\_release\_meal\_skeleton\_seq131\_M1292       | meal       | test     | 3820       | 2735          | 71.60%        |
| Apartment\_release\_work\_skeleton\_seq131\_M1292       | work       | train    | 3803       | 2894          | 76.10%        |
| Apartment\_release\_work\_skeleton\_seq109\_M1292       | work       | train    | 3580       | 2782          | 77.71%        |
| Apartment\_release\_work\_skeleton\_seq110\_M1292       | work       | train    | 3662       | 2760          | 75.37%        |
| Apartment\_release\_decoration\_skeleton\_seq140\_M1292 | decoration | train    | 3604       | 2766          | 76.75%        |
| Apartment\_release\_decoration\_skeleton\_seq137\_M1292 | decoration | train    | 3736       | 2745          | 73.47%        |
| Apartment\_release\_work\_skeleton\_seq136\_M1292       | work       | train    | 3938       | 2741          | 69.60%        |
| Apartment\_release\_meal\_skeleton\_seq136\_M1292       | meal       | train    | 3837       | 2743          | 71.49%        |
| Apartment\_release\_work\_skeleton\_seq106\_M1292       | work       | train    | 4478       | 2723          | 60.81%        |
| Apartment\_release\_meal\_skeleton\_seq134\_M1292       | meal       | train    | 3730       | 2728          | 73.14%        |
| Apartment\_release\_work\_skeleton\_seq134\_M1292       | work       | train    | 3658       | 2743          | 74.99%        |
| Apartment\_release\_decoration\_skeleton\_seq135\_M1292 | decoration | train    | 3605       | 2716          | 75.34%        |
| Apartment\_release\_decoration\_skeleton\_seq138\_M1292 | decoration | train    | 3771       | 2722          | 72.18%        |
| Apartment\_release\_decoration\_skeleton\_seq132\_M1292 | decoration | train    | 3515       | 2747          | 78.15%        |
| Apartment\_release\_work\_skeleton\_seq139\_M1292       | work       | train    | 3766       | 2739          | 72.73%        |
| Apartment\_release\_work\_skeleton\_seq133\_M1292       | work       | train    | 3645       | 2730          | 74.90%        |
| Apartment\_release\_meal\_skeleton\_seq139\_M1292       | meal       | train    | 3778       | 2760          | 73.05%        |
| Apartment\_release\_meal\_skeleton\_seq133\_M1292       | meal       | train    | 3639       | 2712          | 74.53%        |
| Apartment\_release\_work\_skeleton\_seq140\_M1292       | work       | train    | 3659       | 2710          | 74.06%        |
| Apartment\_release\_work\_skeleton\_seq137\_M1292       | work       | train    | 3632       | 2717          | 74.81%        |
| Apartment\_release\_meal\_skeleton\_seq140\_M1292       | meal       | train    | 3633       | 2729          | 75.12%        |
| Apartment\_release\_meal\_skeleton\_seq137\_M1292       | meal       | train    | 3632       | 2732          | 75.22%        |
| Apartment\_release\_decoration\_skeleton\_seq136\_M1292 | decoration | train    | 3750       | 2727          | 72.72%        |
| Apartment\_release\_decoration\_skeleton\_seq131\_M1292 | decoration | train    | 3659       | 2840          | 77.62%        |
| Apartment\_release\_work\_skeleton\_seq108\_M1292       | work       | train    | 3692       | 2802          | 75.89%        |

> 说明：training 列中 1 显示为 train，0 显示为 test，frame\_num 为每个序列的帧数。

---

## 关于 useful frame

在处理 ADT 数据时，并不是所有帧都能成功得到 gaze 投影结果。代码中只有当 gaze 点可以从 **CPF → Camera → Pixels** 的链条投影到相机图像上时，才会认为该帧是 **useful frame**：

* 如果投影失败（例如 gaze 落在视野外、计算出错或无效值），该帧就会被丢弃；
* 这样保证保留下来的都是“有效注视帧”，方便后续训练和可视化。

因此：

* **frame\_num** 表示原始总帧数；
* **useful\_frame** 表示有效帧数；
* **useful\_ratio** = useful\_frame / frame\_num，反映该序列中多少比例的帧可以使用。

在可视化时，依旧保留了原始帧号信息（存储在 `gaze[:,5]` 中），以便回溯和对齐图像帧。

---

如后续数据有新增类别，请及时补充。

## 备注

* 请确保所有序列名称在实际数据目录下均存在。
* 可根据实际需求扩展其他字段。
