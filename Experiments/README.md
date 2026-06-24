

这个目录主要用于实验尝试，包括还没有沉淀为通用 sandbox API 的临时代码、
notebook 和实验报告。生成的表格、图片和缓存默认不进入版本控制。

1. 数据处理：
    - `downsample_processing/downsample_processing_single.py`: 将 30Hz gaze
      处理成对应的频率（15、10、6、5）。

2. SparseGaze evaluation:
    - `sparsegaze_evaluation/REPORT.md`: evaluation plan and metric priorities.
    - `sparsegaze_evaluation/event_evaluation.py`: missing-frame
      event-conditioned angular-error evaluation.
    - `sparsegaze_evaluation/sparsegaze_event_comparison_viewer.ipynb`:
      interactive event timeline viewer.
    - `sparsegaze_evaluation/sparsegaze_overall_evaluation_viewer.ipynb`:
      all-sequence common-frame evaluation viewer.


### `GazeSample`

-  `query_timestamp_ns`: Original query timestamp in device time, nanoseconds. 时间戳
-  `gaze_valid` : Whether the gaze query returned valid data. gaze是否有效
-  `gaze_dt_ns` : Time difference between query and gaze data, nanoseconds.
-  `yaw_rad` : Yaw angle in radians.
-  `pitch_rad`: Pitch angle in radians.
-  `depth_m` : Distance along the CPF gaze ray, meters.
-  `gaze_dir_cpf_unit_x`:  Unit gaze direction X in CPF.
-  `gaze_dir_cpf_unit_y`:
-  `gaze_dir_cpf_unit_z`
-  `yaw_confidence_width_rad`
-  `pitch_confidence_width_rad`
-  `projection_valid`
-  `gaze_u_px`
-  `gaze_v_px`
-  `projection_in_image`
-  `image_width_px`
-  `image_height_px`
-  `pose_valid`
-  `pose_dt_ns`
-  `pose_quality_score`
-  `gaze_origin_scene_x_m`
-  `gaze_origin_scene_y_m`
-  `gaze_origin_scene_z_m`
-  `gaze_point_scene_x_m`
-  `gaze_point_scene_y_m`
-  `gaze_point_scene_z_m`
-  `gaze_dir_scene_unit_x`
-  `gaze_dir_scene_unit_y`
-  `gaze_dir_scene_unit_z`
-  `validation_notes`
