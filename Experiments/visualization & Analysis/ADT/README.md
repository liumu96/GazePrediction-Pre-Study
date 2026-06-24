
这个文档用于记录分析论文模型的结果

先只分析ADT数据集 ---- 后续再扩展到其它数据集

现在有了定量的一些数据支撑，比如MAE和JS

MAE:

| Method | ADT |
| --- | ---: |
| Head direction | 27.48 / 27.49 / 27.48 |
| Repeat | <u>1.67</u> / 2.35 / 3.71 |
| CPF repeat | 1.67 / <u>2.31</u> / <u>3.51</u> |
| SparseGaze | **1.43 / 2.01 / 3.12** |

JS:
| Method | ADT |
| --- | ---: |
| Head direction | 0.868 / 0.390 / 0.206 |
| Repeat | 0.409 / 0.413 / 0.410 |
| CPF repeat | <u>0.050 / 0.050 / 0.050</u> |
| SparseGaze | **0.038 / 0.039 / 0.038** |

备注：JS在egobody、ritw、nymeria上cpf repeat是最好的

和HAGI++的对比

MAE:

| Method | ADT |
| --- | ---: |
| HAGI++ | 1.52 / 2.16 / **3.10** |
| SparseGaze matched | **1.42 / 2.00** / 3.11 |

JS：

| Method | ADT |
| --- | ---: |
| HAGI++ | 0.846 / 0.401 / 0.198 |
| SparseGaze matched | **0.007 / 0.012 / 0.016** |

1. 首先是主要模型的定性结果分析

现在MAE结果显示SparseGaze是最好的，我想对比看看几个baseline

需要选一组用于论文的定性可视化对比

- 一个image scanpath的连续的图，图上绘制出gt和predict，然后还有gt，这样可以对比看predict的效果，展示6Hz的结果吧，把hagi++的结果也加上



2. 关于定量结果分析，现在用的MAE和JS
但是我们有每一帧的depth信息，根据gaze dir就可以求出gaze point， 那那是不是可以求出euclidian distance error呢？



Ablation Study:

1. Head-Motion Input Ablation

| Variant | 15 Hz | 10 Hz | 6 Hz |
| --- | ---: | ---: | ---: |
| No head movement | 1.5122 / 0.0314 | 2.2758 / 0.0372 | 3.7143 / 0.0444 |
| Translation only | 1.4898 / 0.0333 | 2.1849 / 0.0354 | 3.5141 / 0.0402 |
| Rotation only | 1.4433 / 0.0337 | 2.0991 / 0.0299 | 3.2661 / 0.0307 |
| Rotation + translation | **1.4255** / 0.0375 | **2.0074** / 0.0386 | **3.1180** / 0.0384 |

从结果可以看出，head motion帮助很大；其中rotation的作用比translation更大，这一点和hagi++论文是一致的；

但JS结果并没有更好；

2. Gaze

这里探讨的比较多
- 首先研究的是：sparsegaze的帮助有多大

| Dataset | Variant | 15 Hz | 10 Hz | 6 Hz |
| --- | --- | ---: | ---: | ---: |
| ADT | Pose2Gaze-past | 12.0702 | 12.0604 | 12.0608 |
| ADT | Pose2Gaze-present | 11.5493 | 11.5447 | 11.5435 |
| ADT | Head motion only, no gaze history | 10.5589 / 0.5867 | 10.5558 / 0.2304 | 10.5467 / 0.1487 |
| ADT | SparseGaze, sparse gaze + head motion | **1.4255** / **0.0375** | **2.0074** / **0.0386** | **3.1180** / **0.0384** |


有了sparsegaze之后，性能提高了很多

SparseGaze模型去掉gaze之后也比Pose2Gaze好

TODO：补充上Pose2Gaze的JS参数

| Dataset | Anchor setting | 15 Hz | 10 Hz | 6 Hz |
| --- | --- | ---: | ---: | ---: |
| ADT | Sparse anchors + prediction feedback | **1.4255** / **0.0375** | **2.0074** / **0.0386** | **3.1180** / **0.0384** |
| ADT | Prediction feedback only | 16.4411 / 0.0473 | 16.4499 / 0.0475 | 16.4534 / 0.0462 |

| Dataset | Feedback pattern | 15 Hz | 10 Hz | 6 Hz |
| --- | --- | ---: | ---: | ---: |
| ADT | Sparse anchors + prediction feedback | **1.4255** / 0.0375 | **2.0074** / 0.0386 | **3.1180** / 0.0384 |
| ADT | Sparse anchors + repeat feedback | 1.4331 / 0.0348 | 2.1005 / 0.0259 | 3.3300 / 0.0337 |
| ADT | Sparse anchors + extrapolate feedback | 1.4500 / **0.0315** | 2.2977 / **0.0167** | 4.0545 / 0.0549 |



3. residual-over-repeat

| Dataset | Output head | 15 Hz | 10 Hz | 6 Hz |
| --- | --- | ---: | ---: | ---: |
| ADT | Direct | 1.4571 / 0.0353 | 2.0378 / 0.0364 | 3.1539 / 0.0368 |
| ADT | Residual over repeat | **1.4255** / 0.0375 | **2.0074** / 0.0386 | **3.1180** / 0.0384 |

这里要分析一下residual-over-repeat对event和interaction的影响
