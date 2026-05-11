"""SparseGaze-oriented diagnostics for head-motion utility.

The functions in this module treat head motion as a candidate signal for
recovering missing high-frequency gaze samples between sparse gaze anchors.
They are diagnostic tools, not model-training code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import isfinite, sqrt
from typing import Iterable, Sequence

import numpy as np

from .gaze import GazeSample
from .gaze_dynamics import angular_distance_deg, normalize_vector
from .head import HeadSample
from .head_gaze_analysis import (
    pearson_corr,
    rotation_vector_from_matrix,
)
from .head import relative_rotation_prev_to_cur_matrix
from .scene_gaze_events import SceneGazeEventFeatureRow, SceneGazeFrameLabel


@dataclass(frozen=True)
class SparseGazeHeadUtilitySummaryRow:
    sequence_name: str
    anchor_interval_frames: int
    baseline: str
    event_group: str
    sample_count: int
    mean_cpf_residual_deg: float | None
    mean_scene_residual_deg: float | None
    corr_cpf_residual_vs_current_head_rotation_speed: float | None
    corr_scene_residual_vs_current_head_rotation_speed: float | None
    corr_cpf_residual_vs_cumulative_head_rotation_deg: float | None
    corr_scene_residual_vs_cumulative_head_rotation_deg: float | None
    ridge_cpf_r2_gap_only: float | None
    ridge_cpf_r2_current_head: float | None
    ridge_cpf_r2_head_history: float | None
    ridge_scene_r2_gap_only: float | None
    ridge_scene_r2_current_head: float | None
    ridge_scene_r2_head_history: float | None


@dataclass(frozen=True)
class SparseGazeLeadLagRow:
    sequence_name: str
    lag_frames: int
    lag_ms: float | None
    target_name: str
    head_feature_name: str
    sample_count: int
    pearson_corr: float | None


@dataclass(frozen=True)
class _FrameSignals:
    frame_index: int
    timestamp_ns: int
    cpf_dir: tuple[float, float, float] | None
    scene_dir: tuple[float, float, float] | None
    dt_from_prev_s: float | None
    cpf_local_velocity_deg_s: float | None
    scene_velocity_deg_s: float | None
    event_label: str | None
    head_rotation_speed_deg_s: float | None
    head_translation_speed_m_s: float | None
    head_rotvec_x_rad: float | None
    head_rotvec_y_rad: float | None
    head_rotvec_z_rad: float | None
    head_rotation_angle_step_deg: float | None


@dataclass(frozen=True)
class _ResidualFrameRow:
    frame_index: int
    event_label: str | None
    cpf_residual_deg: float | None
    scene_residual_deg: float | None
    gap_fraction: float
    frames_since_anchor: int
    current_head_rotation_speed_deg_s: float | None
    current_head_translation_speed_m_s: float | None
    current_head_rotvec_x_rad: float | None
    current_head_rotvec_y_rad: float | None
    current_head_rotvec_z_rad: float | None
    cumulative_head_rotation_deg: float | None
    history_mean_head_rotation_speed_deg_s: float | None
    history_max_head_rotation_speed_deg_s: float | None
    cumulative_head_rotvec_x_rad: float | None
    cumulative_head_rotvec_y_rad: float | None
    cumulative_head_rotvec_z_rad: float | None


def rows_to_dicts(rows: Iterable[object]) -> list[dict[str, object]]:
    return [asdict(row) for row in rows]


def build_frame_signals(
    gaze_samples: Sequence[GazeSample],
    head_samples: Sequence[HeadSample],
    scene_event_features: Sequence[SceneGazeEventFeatureRow] | None = None,
    scene_frame_labels: Sequence[SceneGazeFrameLabel] | None = None,
) -> list[_FrameSignals]:
    if len(gaze_samples) != len(head_samples):
        raise ValueError(
            f"gaze/head length mismatch: {len(gaze_samples)} gaze vs {len(head_samples)} head"
        )

    scene_velocity_by_frame = {
        row.frame_index: row.scene_velocity_deg_s for row in scene_event_features or []
    }
    scene_label_by_frame = {
        row.frame_index: row.scene_event_label for row in scene_frame_labels or []
    }

    rows: list[_FrameSignals] = []
    prev_gaze: GazeSample | None = None
    prev_cpf_dir: tuple[float, float, float] | None = None
    for frame_index, (gaze, head) in enumerate(zip(gaze_samples, head_samples)):
        cpf_dir = _valid_unit_tuple(gaze.gaze_dir_cpf_unit_xyz)
        scene_dir = _valid_unit_tuple(gaze.gaze_dir_scene_unit_xyz)

        dt_from_prev_s: float | None = None
        if prev_gaze is not None:
            dt_from_prev_s = (gaze.query_timestamp_ns - prev_gaze.query_timestamp_ns) / 1e9
            if dt_from_prev_s <= 0:
                dt_from_prev_s = None

        cpf_velocity = None
        if dt_from_prev_s and prev_cpf_dir is not None and cpf_dir is not None:
            cpf_velocity = angular_distance_deg(prev_cpf_dir, cpf_dir) / dt_from_prev_s

        rotvec = _head_rotvec(head)
        rows.append(
            _FrameSignals(
                frame_index=frame_index,
                timestamp_ns=gaze.query_timestamp_ns,
                cpf_dir=cpf_dir,
                scene_dir=scene_dir,
                dt_from_prev_s=dt_from_prev_s,
                cpf_local_velocity_deg_s=cpf_velocity,
                scene_velocity_deg_s=scene_velocity_by_frame.get(frame_index),
                event_label=scene_label_by_frame.get(frame_index),
                head_rotation_speed_deg_s=_finite_or_none(head.head_rotation_speed_deg_s),
                head_translation_speed_m_s=_finite_or_none(head.head_translation_speed_m_s),
                head_rotvec_x_rad=rotvec[0] if rotvec is not None else None,
                head_rotvec_y_rad=rotvec[1] if rotvec is not None else None,
                head_rotvec_z_rad=rotvec[2] if rotvec is not None else None,
                head_rotation_angle_step_deg=_finite_or_none(
                    head.head_rotation_angle_step_deg
                ),
            )
        )
        prev_gaze = gaze
        prev_cpf_dir = cpf_dir
    return rows


def build_sparse_anchor_utility_rows(
    sequence_name: str,
    gaze_samples: Sequence[GazeSample],
    head_samples: Sequence[HeadSample],
    scene_event_features: Sequence[SceneGazeEventFeatureRow] | None,
    scene_frame_labels: Sequence[SceneGazeFrameLabel] | None,
    anchor_interval_frames: Sequence[int],
    history_frames: int = 5,
) -> list[SparseGazeHeadUtilitySummaryRow]:
    signals = build_frame_signals(
        gaze_samples=gaze_samples,
        head_samples=head_samples,
        scene_event_features=scene_event_features,
        scene_frame_labels=scene_frame_labels,
    )

    rows: list[SparseGazeHeadUtilitySummaryRow] = []
    for interval in anchor_interval_frames:
        if interval < 2:
            raise ValueError(f"anchor interval must be >= 2, got {interval}")
        for baseline in ("hold_last", "linear_interp"):
            residual_rows = _build_residual_rows(
                signals=signals,
                anchor_interval_frames=interval,
                baseline=baseline,
                history_frames=history_frames,
            )
            for event_group in ("all", "fixation", "transition"):
                group_rows = [
                    row
                    for row in residual_rows
                    if event_group == "all" or row.event_label == event_group
                ]
                rows.append(
                    _summarize_residual_rows(
                        sequence_name=sequence_name,
                        anchor_interval_frames=interval,
                        baseline=baseline,
                        event_group=event_group,
                        residual_rows=group_rows,
                    )
                )
    return rows


def build_lead_lag_rows(
    sequence_name: str,
    gaze_samples: Sequence[GazeSample],
    head_samples: Sequence[HeadSample],
    scene_event_features: Sequence[SceneGazeEventFeatureRow] | None,
    scene_frame_labels: Sequence[SceneGazeFrameLabel] | None,
    max_lag_frames: int = 15,
) -> list[SparseGazeLeadLagRow]:
    signals = build_frame_signals(
        gaze_samples=gaze_samples,
        head_samples=head_samples,
        scene_event_features=scene_event_features,
        scene_frame_labels=scene_frame_labels,
    )
    median_dt_s = _median_dt_s(signals)

    target_series = {
        "cpf_local_velocity_deg_s": [
            row.cpf_local_velocity_deg_s for row in signals
        ],
        "scene_velocity_deg_s": [row.scene_velocity_deg_s for row in signals],
        "scene_transition_indicator": [
            1.0 if row.event_label == "transition" else 0.0
            if row.event_label is not None
            else None
            for row in signals
        ],
    }
    head_series = {
        "head_rotation_speed_deg_s": [
            row.head_rotation_speed_deg_s for row in signals
        ],
        "head_translation_speed_m_s": [
            row.head_translation_speed_m_s for row in signals
        ],
    }

    rows: list[SparseGazeLeadLagRow] = []
    for lag in range(-max_lag_frames, max_lag_frames + 1):
        lag_ms = lag * median_dt_s * 1000 if median_dt_s is not None else None
        for target_name, target_values in target_series.items():
            for head_feature_name, head_values in head_series.items():
                head_pairs, target_pairs = _lagged_pairs(head_values, target_values, lag)
                rows.append(
                    SparseGazeLeadLagRow(
                        sequence_name=sequence_name,
                        lag_frames=lag,
                        lag_ms=lag_ms,
                        target_name=target_name,
                        head_feature_name=head_feature_name,
                        sample_count=len(head_pairs),
                        pearson_corr=pearson_corr(head_pairs, target_pairs),
                    )
                )
    return rows


def aggregate_summary_rows(
    rows: Sequence[SparseGazeHeadUtilitySummaryRow],
) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, str], list[SparseGazeHeadUtilitySummaryRow]] = {}
    for row in rows:
        key = (row.anchor_interval_frames, row.baseline, row.event_group)
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    metric_names = [
        "sample_count",
        "mean_cpf_residual_deg",
        "mean_scene_residual_deg",
        "corr_cpf_residual_vs_current_head_rotation_speed",
        "corr_scene_residual_vs_current_head_rotation_speed",
        "corr_cpf_residual_vs_cumulative_head_rotation_deg",
        "corr_scene_residual_vs_cumulative_head_rotation_deg",
        "ridge_cpf_r2_gap_only",
        "ridge_cpf_r2_current_head",
        "ridge_cpf_r2_head_history",
        "ridge_scene_r2_gap_only",
        "ridge_scene_r2_current_head",
        "ridge_scene_r2_head_history",
    ]
    for (interval, baseline, event_group), group_rows in sorted(groups.items()):
        out: dict[str, object] = {
            "anchor_interval_frames": interval,
            "baseline": baseline,
            "event_group": event_group,
            "sequence_count": len(group_rows),
        }
        for metric_name in metric_names:
            values = [
                float(getattr(row, metric_name))
                for row in group_rows
                if _is_finite_number(getattr(row, metric_name))
            ]
            out[f"{metric_name}_mean"] = float(np.mean(values)) if values else None
            out[f"{metric_name}_median"] = (
                float(np.median(values)) if values else None
            )
        aggregate_rows.append(out)
    return aggregate_rows


def aggregate_lead_lag_rows(rows: Sequence[SparseGazeLeadLagRow]) -> list[dict[str, object]]:
    groups: dict[tuple[int, str, str], list[SparseGazeLeadLagRow]] = {}
    for row in rows:
        key = (row.lag_frames, row.target_name, row.head_feature_name)
        groups.setdefault(key, []).append(row)

    aggregate_rows: list[dict[str, object]] = []
    for (lag, target_name, head_feature_name), group_rows in sorted(groups.items()):
        corrs = [
            float(row.pearson_corr)
            for row in group_rows
            if _is_finite_number(row.pearson_corr)
        ]
        sample_counts = [
            row.sample_count for row in group_rows if _is_finite_number(row.sample_count)
        ]
        lag_ms_values = [
            float(row.lag_ms) for row in group_rows if _is_finite_number(row.lag_ms)
        ]
        aggregate_rows.append(
            {
                "lag_frames": lag,
                "lag_ms_mean": float(np.mean(lag_ms_values)) if lag_ms_values else None,
                "target_name": target_name,
                "head_feature_name": head_feature_name,
                "sequence_count": len(group_rows),
                "sample_count_sum": int(sum(sample_counts)),
                "pearson_corr_mean": float(np.mean(corrs)) if corrs else None,
                "pearson_corr_median": float(np.median(corrs)) if corrs else None,
            }
        )
    return aggregate_rows


def _build_residual_rows(
    signals: Sequence[_FrameSignals],
    anchor_interval_frames: int,
    baseline: str,
    history_frames: int,
) -> list[_ResidualFrameRow]:
    rows: list[_ResidualFrameRow] = []
    for frame_index, signal in enumerate(signals):
        prev_anchor = (frame_index // anchor_interval_frames) * anchor_interval_frames
        if prev_anchor == frame_index:
            continue
        next_anchor = prev_anchor + anchor_interval_frames
        if prev_anchor < 0 or prev_anchor >= len(signals):
            continue
        if baseline == "linear_interp" and next_anchor >= len(signals):
            continue

        pred_cpf = _predict_direction(
            signals=signals,
            direction_name="cpf_dir",
            frame_index=frame_index,
            prev_anchor=prev_anchor,
            next_anchor=next_anchor,
            anchor_interval_frames=anchor_interval_frames,
            baseline=baseline,
        )
        pred_scene = _predict_direction(
            signals=signals,
            direction_name="scene_dir",
            frame_index=frame_index,
            prev_anchor=prev_anchor,
            next_anchor=next_anchor,
            anchor_interval_frames=anchor_interval_frames,
            baseline=baseline,
        )
        cpf_residual = _angular_residual(pred_cpf, signal.cpf_dir)
        scene_residual = _angular_residual(pred_scene, signal.scene_dir)
        if cpf_residual is None and scene_residual is None:
            continue

        history_start = max(prev_anchor + 1, frame_index - history_frames + 1)
        history_slice = signals[history_start : frame_index + 1]
        since_anchor_slice = signals[prev_anchor + 1 : frame_index + 1]
        rows.append(
            _ResidualFrameRow(
                frame_index=frame_index,
                event_label=signal.event_label,
                cpf_residual_deg=cpf_residual,
                scene_residual_deg=scene_residual,
                gap_fraction=(frame_index - prev_anchor) / anchor_interval_frames,
                frames_since_anchor=frame_index - prev_anchor,
                current_head_rotation_speed_deg_s=signal.head_rotation_speed_deg_s,
                current_head_translation_speed_m_s=signal.head_translation_speed_m_s,
                current_head_rotvec_x_rad=signal.head_rotvec_x_rad,
                current_head_rotvec_y_rad=signal.head_rotvec_y_rad,
                current_head_rotvec_z_rad=signal.head_rotvec_z_rad,
                cumulative_head_rotation_deg=_sum_optional(
                    row.head_rotation_angle_step_deg for row in since_anchor_slice
                ),
                history_mean_head_rotation_speed_deg_s=_mean_optional(
                    row.head_rotation_speed_deg_s for row in history_slice
                ),
                history_max_head_rotation_speed_deg_s=_max_optional(
                    row.head_rotation_speed_deg_s for row in history_slice
                ),
                cumulative_head_rotvec_x_rad=_sum_optional(
                    row.head_rotvec_x_rad for row in since_anchor_slice
                ),
                cumulative_head_rotvec_y_rad=_sum_optional(
                    row.head_rotvec_y_rad for row in since_anchor_slice
                ),
                cumulative_head_rotvec_z_rad=_sum_optional(
                    row.head_rotvec_z_rad for row in since_anchor_slice
                ),
            )
        )
    return rows


def _summarize_residual_rows(
    sequence_name: str,
    anchor_interval_frames: int,
    baseline: str,
    event_group: str,
    residual_rows: Sequence[_ResidualFrameRow],
) -> SparseGazeHeadUtilitySummaryRow:
    cpf_residuals = [row.cpf_residual_deg for row in residual_rows]
    scene_residuals = [row.scene_residual_deg for row in residual_rows]
    head_rot = [row.current_head_rotation_speed_deg_s for row in residual_rows]
    cumulative_head_rot = [row.cumulative_head_rotation_deg for row in residual_rows]

    return SparseGazeHeadUtilitySummaryRow(
        sequence_name=sequence_name,
        anchor_interval_frames=anchor_interval_frames,
        baseline=baseline,
        event_group=event_group,
        sample_count=len(residual_rows),
        mean_cpf_residual_deg=_mean_optional(cpf_residuals),
        mean_scene_residual_deg=_mean_optional(scene_residuals),
        corr_cpf_residual_vs_current_head_rotation_speed=_corr_optional(
            cpf_residuals, head_rot
        ),
        corr_scene_residual_vs_current_head_rotation_speed=_corr_optional(
            scene_residuals, head_rot
        ),
        corr_cpf_residual_vs_cumulative_head_rotation_deg=_corr_optional(
            cpf_residuals, cumulative_head_rot
        ),
        corr_scene_residual_vs_cumulative_head_rotation_deg=_corr_optional(
            scene_residuals, cumulative_head_rot
        ),
        ridge_cpf_r2_gap_only=_blocked_ridge_r2(
            residual_rows, target_name="cpf_residual_deg", feature_set="gap_only"
        ),
        ridge_cpf_r2_current_head=_blocked_ridge_r2(
            residual_rows, target_name="cpf_residual_deg", feature_set="current_head"
        ),
        ridge_cpf_r2_head_history=_blocked_ridge_r2(
            residual_rows, target_name="cpf_residual_deg", feature_set="head_history"
        ),
        ridge_scene_r2_gap_only=_blocked_ridge_r2(
            residual_rows, target_name="scene_residual_deg", feature_set="gap_only"
        ),
        ridge_scene_r2_current_head=_blocked_ridge_r2(
            residual_rows, target_name="scene_residual_deg", feature_set="current_head"
        ),
        ridge_scene_r2_head_history=_blocked_ridge_r2(
            residual_rows, target_name="scene_residual_deg", feature_set="head_history"
        ),
    )


def _blocked_ridge_r2(
    rows: Sequence[_ResidualFrameRow],
    target_name: str,
    feature_set: str,
    fold_count: int = 5,
    ridge_alpha: float = 1.0,
) -> float | None:
    matrix_rows: list[list[float]] = []
    targets: list[float] = []
    for row in rows:
        target = getattr(row, target_name)
        features = _feature_values(row, feature_set)
        if not _is_finite_number(target):
            continue
        if features is None or not all(_is_finite_number(value) for value in features):
            continue
        matrix_rows.append([float(value) for value in features])
        targets.append(float(target))

    if len(targets) < max(20, fold_count * 3):
        return None

    x = np.asarray(matrix_rows, dtype=float)
    y = np.asarray(targets, dtype=float)
    if float(np.var(y)) <= 1e-12:
        return None

    indices = np.arange(len(y))
    folds = [fold for fold in np.array_split(indices, fold_count) if len(fold) > 0]
    predictions = np.full_like(y, fill_value=np.nan, dtype=float)

    for test_indices in folds:
        train_mask = np.ones(len(y), dtype=bool)
        train_mask[test_indices] = False
        train_indices = indices[train_mask]
        if len(train_indices) < 2:
            continue

        x_train = x[train_indices]
        y_train = y[train_indices]
        x_test = x[test_indices]

        mean = np.mean(x_train, axis=0)
        std = np.std(x_train, axis=0)
        std[std < 1e-12] = 1.0
        x_train_std = (x_train - mean) / std
        x_test_std = (x_test - mean) / std

        x_train_aug = np.column_stack([np.ones(len(x_train_std)), x_train_std])
        x_test_aug = np.column_stack([np.ones(len(x_test_std)), x_test_std])
        penalty = np.eye(x_train_aug.shape[1])
        penalty[0, 0] = 0.0
        coefficients = np.linalg.pinv(
            x_train_aug.T @ x_train_aug + ridge_alpha * penalty
        ) @ x_train_aug.T @ y_train
        predictions[test_indices] = x_test_aug @ coefficients

    valid = np.isfinite(predictions)
    if int(np.sum(valid)) < max(10, fold_count):
        return None
    y_valid = y[valid]
    pred_valid = predictions[valid]
    denominator = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
    if denominator <= 1e-12:
        return None
    numerator = float(np.sum((y_valid - pred_valid) ** 2))
    return 1.0 - numerator / denominator


def _feature_values(row: _ResidualFrameRow, feature_set: str) -> list[float] | None:
    gap_features = [
        row.gap_fraction,
        float(row.frames_since_anchor),
    ]
    current_head_features = gap_features + [
        row.current_head_rotation_speed_deg_s,
        row.current_head_translation_speed_m_s,
        row.current_head_rotvec_x_rad,
        row.current_head_rotvec_y_rad,
        row.current_head_rotvec_z_rad,
    ]
    head_history_features = current_head_features + [
        row.cumulative_head_rotation_deg,
        row.history_mean_head_rotation_speed_deg_s,
        row.history_max_head_rotation_speed_deg_s,
        row.cumulative_head_rotvec_x_rad,
        row.cumulative_head_rotvec_y_rad,
        row.cumulative_head_rotvec_z_rad,
    ]
    if feature_set == "gap_only":
        return gap_features
    if feature_set == "current_head":
        return current_head_features
    if feature_set == "head_history":
        return head_history_features
    raise ValueError(f"unknown feature set: {feature_set}")


def _predict_direction(
    signals: Sequence[_FrameSignals],
    direction_name: str,
    frame_index: int,
    prev_anchor: int,
    next_anchor: int,
    anchor_interval_frames: int,
    baseline: str,
) -> tuple[float, float, float] | None:
    prev_dir = getattr(signals[prev_anchor], direction_name)
    if baseline == "hold_last":
        return prev_dir
    if baseline != "linear_interp":
        raise ValueError(f"unknown baseline: {baseline}")
    next_dir = getattr(signals[next_anchor], direction_name)
    if prev_dir is None or next_dir is None:
        return None
    alpha = (frame_index - prev_anchor) / anchor_interval_frames
    return _lerp_unit(prev_dir, next_dir, alpha)


def _lerp_unit(
    first: Sequence[float], second: Sequence[float], alpha: float
) -> tuple[float, float, float] | None:
    vec = tuple((1.0 - alpha) * first[i] + alpha * second[i] for i in range(3))
    return normalize_vector(vec)


def _angular_residual(
    predicted: Sequence[float] | None, target: Sequence[float] | None
) -> float | None:
    if predicted is None or target is None:
        return None
    return angular_distance_deg(predicted, target)


def _lagged_pairs(
    head_values: Sequence[float | None],
    target_values: Sequence[float | None],
    lag_frames: int,
) -> tuple[list[float], list[float]]:
    head_pairs: list[float] = []
    target_pairs: list[float] = []
    for head_index, head_value in enumerate(head_values):
        target_index = head_index + lag_frames
        if target_index < 0 or target_index >= len(target_values):
            continue
        target_value = target_values[target_index]
        if _is_finite_number(head_value) and _is_finite_number(target_value):
            head_pairs.append(float(head_value))
            target_pairs.append(float(target_value))
    return head_pairs, target_pairs


def _head_rotvec(head: HeadSample) -> tuple[float, float, float] | None:
    rotation = relative_rotation_prev_to_cur_matrix(head)
    if rotation is None:
        return None
    rotvec = rotation_vector_from_matrix(rotation)
    if not all(_is_finite_number(value) for value in rotvec):
        return None
    return tuple(float(value) for value in rotvec)


def _valid_unit_tuple(values: Sequence[float]) -> tuple[float, float, float] | None:
    if len(values) != 3 or not all(_is_finite_number(value) for value in values):
        return None
    norm = sqrt(sum(float(value) * float(value) for value in values))
    if norm <= 1e-12:
        return None
    return tuple(float(value) / norm for value in values)


def _median_dt_s(rows: Sequence[_FrameSignals]) -> float | None:
    values = [
        row.dt_from_prev_s for row in rows if _is_finite_number(row.dt_from_prev_s)
    ]
    if not values:
        return None
    return float(np.median(values))


def _mean_optional(values: Iterable[float | None]) -> float | None:
    finite_values = [float(value) for value in values if _is_finite_number(value)]
    if not finite_values:
        return None
    return float(np.mean(finite_values))


def _max_optional(values: Iterable[float | None]) -> float | None:
    finite_values = [float(value) for value in values if _is_finite_number(value)]
    if not finite_values:
        return None
    return float(max(finite_values))


def _sum_optional(values: Iterable[float | None]) -> float | None:
    finite_values = [float(value) for value in values if _is_finite_number(value)]
    if not finite_values:
        return None
    return float(sum(finite_values))


def _corr_optional(
    first_values: Sequence[float | None], second_values: Sequence[float | None]
) -> float | None:
    first: list[float] = []
    second: list[float] = []
    for first_value, second_value in zip(first_values, second_values):
        if _is_finite_number(first_value) and _is_finite_number(second_value):
            first.append(float(first_value))
            second.append(float(second_value))
    return pearson_corr(first, second)


def _finite_or_none(value: float | None) -> float | None:
    if _is_finite_number(value):
        return float(value)
    return None


def _is_finite_number(value: object) -> bool:
    if value is None:
        return False
    try:
        return isfinite(float(value))
    except (TypeError, ValueError):
        return False
