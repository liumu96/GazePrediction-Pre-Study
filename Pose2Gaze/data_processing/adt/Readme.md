# ADT 数据处理说明

本项目处理的数据源为 `adt.csv`，该文件包含每个数据序列的基本信息。字段及格式如下：

## 数据格式说明

`adt.csv` 为逗号分隔的表格文件，每一行对应一个数据序列。字段如下：

| 字段名         | 说明                                 |
|----------------|--------------------------------------|
| sequence_name  | 数据序列名称或唯一标识               |
| training       | 是否用于训练（1为训练，0为测试）      |
| action         | 行为标签或动作类别                   |

## 示例

```csv
sequence_name,training,action
work_skeleton_seq132_M1292,0,work
meal_skeleton_seq132_M1292,0,meal
decoration_skeleton_seq133_M1292,0,decoration
work_skeleton_seq131_M1292,1,work
```

## 字段说明

- **sequence_name**：每条数据的序列名称或唯一标识，通常对应一个数据文件或文件夹。
- **training**：标记该序列是否用于训练（1）或测试（0）。
- **action**：该序列对应的动作或行为标签。

## action 字段类别及数据统计

根据当前数据，`action` 字段包含以下类别及数量：

| action      | 训练集数量 | 测试集数量 |
|-------------|------------|------------|
| work        | 11         | 4          |
| meal        | 6          | 3          |
| decoration  | 7          | 3          |

## 每个序列详细统计

| sequence_name                                   | action      | training | frame_num |
|-------------------------------------------------|-------------|----------|-----------|
| work_skeleton_seq132_M1292     | work        | test     | 3628      |
| work_skeleton_seq138_M1292     | work        | test     | 3755      |
| meal_skeleton_seq132_M1292     | meal        | test     | 3674      |
| decoration_skeleton_seq133_M1292| decoration  | test     | 3483      |
| decoration_skeleton_seq139_M1292| decoration  | test     | 4102      |
| decoration_skeleton_seq134_M1292| decoration  | test     | 3622      |
| work_skeleton_seq107_M1292     | work        | test     | 3723      |
| meal_skeleton_seq135_M1292     | meal        | test     | 3853      |
| work_skeleton_seq135_M1292     | work        | train    | 3610      |
| meal_skeleton_seq131_M1292     | meal        | train    | 3820      |
| work_skeleton_seq131_M1292     | work        | train    | 3803      |
| work_skeleton_seq109_M1292     | work        | train    | 3580      |
| work_skeleton_seq110_M1292     | work        | train    | 3662      |
| decoration_skeleton_seq140_M1292| decoration  | train    | 3604      |
| decoration_skeleton_seq137_M1292| decoration  | train    | 3736      |
| work_skeleton_seq136_M1292     | work        | train    | 3938      |
| meal_skeleton_seq136_M1292     | meal        | train    | 3837      |
| work_skeleton_seq106_M1292     | work        | train    | 4478      |
| meal_skeleton_seq134_M1292     | meal        | train    | 3730      |
| work_skeleton_seq134_M1292     | work        | train    | 3658      |
| decoration_skeleton_seq135_M1292| decoration  | train    | 3605      |
| decoration_skeleton_seq138_M1292| decoration  | train    | 3771      |
| decoration_skeleton_seq132_M1292| decoration  | train    | 3515      |
| work_skeleton_seq139_M1292     | work        | train    | 3766      |
| work_skeleton_seq133_M1292     | work        | train    | 3645      |
| meal_skeleton_seq139_M1292     | meal        | train    | 3778      |
| meal_skeleton_seq133_M1292     | meal        | train    | 3639      |
| work_skeleton_seq140_M1292     | work        | train    | 3659      |
| work_skeleton_seq137_M1292     | work        | train    | 3632      |
| meal_skeleton_seq140_M1292     | meal        | train    | 3633      |
| meal_skeleton_seq137_M1292     | meal        | train    | 3632      |
| decoration_skeleton_seq136_M1292| decoration  | train    | 3750      |
| decoration_skeleton_seq131_M1292| decoration  | train    | 3659      |
| work_skeleton_seq108_M1292     | work        | train    | 3692      |

> 说明：training 列中 1 显示为 train，0 显示为 test，frame_num 为每个序列的帧数。

---


如后续数据有新增类别，请及时补充。

## 备注

- 请确保所有序列名称在实际数据目录下均存在。
- 可根据实际需求扩展其他字段。