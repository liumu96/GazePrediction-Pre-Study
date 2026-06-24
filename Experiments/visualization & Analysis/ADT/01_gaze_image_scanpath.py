#!/usr/bin/env python
"""Draw ADT ground-truth and optional SparseGaze image-space scanpaths.

The first use case is ground-truth qualitative inspection:

    python "Experiments/visualization & Analysis/ADT/01_gaze_image_scanpath.py" \
      Apartment_release_decoration_skeleton_seq133_M1292 \
      --start-frame 0 --end-frame 120 --stride 4

      conda run -n adt python "Experiments/visualization & Analysis/ADT/01_gaze_image_scanpath.py" \
  Apartment_release_decoration_skeleton_seq133_M1292 \
  --prediction-npz /home/liumu/Github_Projects/SparseGaze/outputs/eval/adt/sparsegaze/test/rollout/sequence_predictions/Apartment_release_decoration_skeleton_seq133_M1292/hz6_phase0.npz \
  --start-frame 0 \
  --end-frame 120 \
  --stride 5 \
  --tail-frames 30 \
  --run-name sparsegaze_rollout_hz6_frames_0_120_stride5_tail30

SparseGaze prediction NPZ files normally store gaze directions, not RGB image
coordinates.  When ``--prediction-npz`` is provided this script aligns the
prediction rows to the extracted GT gaze rows by timestamp when available, and
falls back to row/frame index alignment otherwise.  It then projects the
prediction ray endpoints back into RGB image space.

For videos, scanpath tails are dynamic: for each current frame ``t``, gaze
points from ``t - tail_frames`` through ``t`` are reprojected into frame ``t``.
The fixed ``*_reference_frame_*`` PNGs are kept only as single-view diagnostics.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

def find_repo_root() -> Path:
    path = Path(__file__).resolve()
    for parent in [path.parent, *path.parents]:
        if (parent / "src").exists() and (parent / "Experiments").exists():
            return parent
    return Path(__file__).resolve().parents[3]


REPO_ROOT = find_repo_root()
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from adt_sandbox.config import load_dotenv  # noqa: E402
from adt_sandbox.gaze import (  # noqa: E402
    GazeSample,
    get_rgb_image,
    project_scene_points_to_rgb,
    read_samples_csv,
)
from adt_sandbox.providers import create_adt_providers  # noqa: E402
from adt_sandbox.results import find_sequence_file, reports_root  # noqa: E402
from visualization.gaze_outputs import (  # noqa: E402
    downsample_pairs,
    load_visualization_context,
    reference_scanpath_from_samples,
    slice_items,
    write_reference_frame_scanpath_clean,
    write_reference_frame_scanpath_overlay,
    write_video_from_overlay_frames,
)
from visualization.prediction_eval import normalize_vectors  # noqa: E402

load_dotenv(REPO_ROOT / ".env")

DEFAULT_HAGI_NPZ = Path(
    "/home/liumu/Github_Projects/HAGI/save/head/hagi++_imputation/"
    "adt_low_framerate_sliding/sliding_primary_nsample20_framerate_6.npz"
)
DEFAULT_HAGI_ADT_DATA = Path(
    "/home/liumu/Github_Projects/HAGI/datasets/adt/gaze_head_adt.npy"
)
TRACK_COLORS = {
    "GT": "#111111",
    "SparseGaze": "#ff7f0e",
    "HAGI++": "#2ca02c",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "sequence",
        help="ADT sequence id resolved under ADT_DATA_ROOT, or an absolute sequence path.",
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=None,
        help="Explicit ground-truth gaze_samples.csv path. Overrides --reports-dir lookup.",
    )
    parser.add_argument(
        "--reports-dir",
        type=Path,
        default=None,
        help=(
            "Structured reports root. Defaults to REPORTS_DIR, then "
            "/mnt/d/SparseGaze/ADT-Gaze-structured."
        ),
    )
    parser.add_argument("--start-frame", type=int, default=0, help="Inclusive GT row/frame start.")
    parser.add_argument("--end-frame", type=int, default=120, help="Exclusive GT row/frame end.")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth frame in the selected window. The last frame is preserved.",
    )
    parser.add_argument(
        "--tail-frames",
        type=int,
        default=30,
        help=(
            "Dynamic scanpath tail length in original GT row/frame units. "
            "Each video frame reprojects gaze points from current-tail..current "
            "into the current RGB frame."
        ),
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional output subdirectory name.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "outputs" / "figures" / "gaze_image_scanpath",
        help="Directory for generated figures.",
    )
    parser.add_argument(
        "--prediction-npz",
        type=Path,
        default=None,
        help="Optional SparseGaze per-sequence prediction NPZ.",
    )
    parser.add_argument(
        "--pred-key",
        default="pred_xyz",
        help="Prediction direction array key inside --prediction-npz.",
    )
    parser.add_argument(
        "--depth-mode",
        choices=["gt_depth", "fixed"],
        default="gt_depth",
        help="Depth used to place prediction ray endpoints before RGB projection.",
    )
    parser.add_argument(
        "--fixed-depth-m",
        type=float,
        default=2.0,
        help="Prediction ray endpoint distance when --depth-mode fixed or GT depth is missing.",
    )
    parser.add_argument(
        "--hagi-npz",
        type=Path,
        default=None,
        help=(
            "Optional HAGI++ sliding_primary NPZ. Example default path: "
            f"{DEFAULT_HAGI_NPZ}"
        ),
    )
    parser.add_argument(
        "--hagi-adt-data",
        type=Path,
        default=DEFAULT_HAGI_ADT_DATA,
        help="HAGI++ ADT cache containing T_world_CPF and timestamps.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Skip packing per-frame overlays into mp4 videos.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = resolve_reports_dir(args.reports_dir)
    sequence_name = Path(args.sequence).name
    csv_path = args.input_csv or find_gaze_samples_csv(reports_dir, sequence_name)
    samples = read_samples_csv(csv_path)

    window_pairs = slice_items(list(enumerate(samples)), args.start_frame, args.end_frame)
    viz_pairs = downsample_pairs(window_pairs, args.stride, include_last=True)
    viz_orders = [row_index for row_index, _ in viz_pairs]
    viz_samples = [sample for _, sample in viz_pairs]
    stream_id_value, make_upright = load_visualization_context(csv_path)

    providers = None
    provider_error: str | None = None
    try:
        providers = create_adt_providers(args.sequence, skeleton_flag=False)
    except RuntimeError as exc:
        provider_error = str(exc)
        if args.prediction_npz is not None or args.hagi_npz is not None:
            raise RuntimeError(
                "Prediction projection requires the ADT provider for RGB calibration and poses, "
                "but provider initialization failed."
            ) from exc

    sequence_output_name = providers.sequence_path.name if providers is not None else sequence_name
    run_name = args.run_name or default_run_name(args.start_frame, args.end_frame, args.stride)
    output_dir = args.output_root / sequence_output_name / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    gt_result = write_gt_scanpath_outputs(
        gt_provider=providers.gt_provider if providers is not None else None,
        output_dir=output_dir,
        viz_pairs=viz_pairs,
        viz_orders=viz_orders,
        viz_samples=viz_samples,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
        write_video=not args.no_video,
        tail_frames=args.tail_frames,
    )

    prediction_result: dict[str, Any] | None = None
    pred_pairs: PredictionPairs | None = None
    if args.prediction_npz is not None:
        pred_pairs = prediction_pairs_from_npz(
            gt_provider=providers.gt_provider,
            npz_path=args.prediction_npz,
            pred_key=args.pred_key,
            gt_pairs=viz_pairs,
            all_gt_samples=samples,
            depth_mode=args.depth_mode,
            fixed_depth_m=args.fixed_depth_m,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
        )
        prediction_result = write_prediction_scanpath_outputs(
            gt_provider=providers.gt_provider if providers is not None else None,
            output_dir=output_dir,
            pred_key=args.pred_key,
            npz_path=args.prediction_npz,
            pred_pairs=pred_pairs,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
            write_video=not args.no_video,
            tail_frames=args.tail_frames,
        )

    hagi_result: dict[str, Any] | None = None
    hagi_pairs: PredictionPairs | None = None
    if args.hagi_npz is not None:
        hagi_pairs = hagi_pairs_from_npz(
            gt_provider=providers.gt_provider,
            hagi_npz=args.hagi_npz,
            hagi_adt_data=args.hagi_adt_data,
            sequence_name=sequence_output_name,
            gt_pairs=viz_pairs,
            all_gt_samples=samples,
            depth_mode=args.depth_mode,
            fixed_depth_m=args.fixed_depth_m,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
        )
        hagi_result = write_hagi_scanpath_outputs(
            gt_provider=providers.gt_provider if providers is not None else None,
            output_dir=output_dir,
            hagi_npz=args.hagi_npz,
            hagi_pairs=hagi_pairs,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
            write_video=not args.no_video,
            tail_frames=args.tail_frames,
        )

    comparison_result: dict[str, Any] | None = None
    if providers is not None and (pred_pairs is not None or hagi_pairs is not None):
        comparison_result = write_comparison_dynamic_tail_outputs(
            gt_provider=providers.gt_provider,
            output_dir=output_dir,
            reference_pairs=viz_pairs,
            pred_pairs=pred_pairs,
            hagi_pairs=hagi_pairs,
            tail_frames=args.tail_frames,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
            write_video=not args.no_video,
        )

    summary = {
        "sequence": sequence_output_name,
        "sequence_path": str(providers.sequence_path) if providers is not None else args.sequence,
        "provider_available": providers is not None,
        "provider_error": provider_error,
        "csv_path": str(csv_path),
        "reports_dir": str(reports_dir),
        "output_dir": str(output_dir),
        "start_frame": args.start_frame,
        "end_frame": args.end_frame,
        "stride": args.stride,
        "tail_frames": args.tail_frames,
        "selected_frames": len(window_pairs),
        "visualized_frames": len(viz_pairs),
        "image_orientation": "upright" if make_upright else "raw",
        "gt": gt_result,
        "prediction": prediction_result,
        "hagi": hagi_result,
        "comparison": comparison_result,
    }
    write_json(output_dir / "summary.json", summary)

    print(f"sequence: {sequence_output_name}")
    print(f"gt_csv: {csv_path}")
    print(f"frames: {args.start_frame}..{args.end_frame} stride={args.stride}")
    print(f"visualized_frames: {len(viz_pairs)}")
    print(f"figures: {output_dir}")
    if provider_error is not None:
        print(f"provider_warning: {provider_error}")
    if prediction_result is not None:
        print(
            "prediction: "
            f"{prediction_result['projected_samples']} projected samples "
            f"({prediction_result['alignment_mode']} alignment)"
        )
    if hagi_result is not None:
        print(
            "hagi++: "
            f"{hagi_result['projected_samples']} projected samples "
            f"({hagi_result['alignment_mode']} alignment)"
        )
    if comparison_result is not None:
        print(f"comparison_video: {comparison_result['dynamic_tail_video']}")


def write_gt_scanpath_outputs(
    *,
    gt_provider: Any | None,
    output_dir: Path,
    viz_pairs: list[tuple[int, GazeSample]],
    viz_orders: list[int],
    viz_samples: list[GazeSample],
    stream_id_value: str,
    make_upright: bool,
    write_video: bool,
    tail_frames: int,
) -> dict[str, Any]:
    """Write ground-truth scanpath figures in RGB image space."""

    direct_path = output_dir / "gt_image_space_scanpath_overlay.png"
    direct_count = write_direct_image_space_scanpath(
        direct_path,
        gt_provider=gt_provider,
        indexed_samples=viz_pairs,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
        title="GT image-space scanpath",
    )

    reference_overlay_path = output_dir / "gt_reference_frame_scanpath_overlay.png"
    reference_clean_path = output_dir / "gt_reference_frame_scanpath_clean.png"
    dynamic_tail_video_path = output_dir / "gt_dynamic_tail_video.mp4"
    dynamic_tail_result: dict[str, Any] | None = None
    if gt_provider is not None:
        reference_scanpath = reference_scanpath_from_samples(
            gt_provider,
            viz_samples,
            viz_orders,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
        )
        write_reference_frame_scanpath_overlay(reference_overlay_path, reference_scanpath)
        write_reference_frame_scanpath_clean(reference_clean_path, reference_scanpath)

        dynamic_tail_result = write_dynamic_tail_overlay_frames(
            gt_provider,
            viz_pairs,
            output_dir / "gt_dynamic_tail_overlays",
            title_prefix="GT dynamic scanpath tail",
            tail_frames=tail_frames,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
        )
        if write_video:
            write_video_from_overlay_frames(
                dynamic_tail_video_path,
                [Path(path) for path in dynamic_tail_result["frame_paths"]],
            )

    return {
        "direct_image_space_scanpath": str(direct_path),
        "direct_image_space_points": direct_count,
        "reference_frame_scanpath_overlay": str(reference_overlay_path) if gt_provider else None,
        "reference_frame_scanpath_clean": str(reference_clean_path) if gt_provider else None,
        "dynamic_tail": dynamic_tail_result,
        "dynamic_tail_video": (
            str(dynamic_tail_video_path)
            if write_video and dynamic_tail_result and dynamic_tail_result["frame_paths"]
            else None
        ),
    }


def write_prediction_scanpath_outputs(
    *,
    gt_provider: Any | None,
    output_dir: Path,
    pred_key: str,
    npz_path: Path,
    pred_pairs: PredictionPairs,
    stream_id_value: str,
    make_upright: bool,
    write_video: bool,
    tail_frames: int,
) -> dict[str, Any]:
    """Write image-space scanpath outputs for aligned prediction samples."""

    if gt_provider is None:
        raise ValueError("Prediction scanpath projection requires an ADT provider.")
    if not pred_pairs.indexed_samples:
        raise ValueError("No prediction samples could be aligned to the selected GT window.")

    pred_dir = output_dir / "prediction"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_samples = [sample for _, sample in pred_pairs.indexed_samples]
    pred_orders = [index for index, _ in pred_pairs.indexed_samples]

    direct_path = pred_dir / "pred_image_space_scanpath_overlay.png"
    direct_count = write_direct_image_space_scanpath(
        direct_path,
        gt_provider=gt_provider,
        indexed_samples=pred_pairs.indexed_samples,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
        title=f"SparseGaze image-space scanpath ({pred_key})",
    )

    reference_scanpath = reference_scanpath_from_samples(
        gt_provider,
        pred_samples,
        pred_orders,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    write_reference_frame_scanpath_overlay(
        pred_dir / "pred_reference_frame_scanpath_overlay.png",
        reference_scanpath,
    )
    write_reference_frame_scanpath_clean(
        pred_dir / "pred_reference_frame_scanpath_clean.png",
        reference_scanpath,
    )

    dynamic_tail_result = write_dynamic_tail_overlay_frames(
        gt_provider,
        pred_pairs.indexed_samples,
        pred_dir / "pred_dynamic_tail_overlays",
        title_prefix=f"SparseGaze dynamic scanpath tail ({pred_key})",
        tail_frames=tail_frames,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    video_path = pred_dir / "pred_dynamic_tail_video.mp4"
    if write_video:
        write_video_from_overlay_frames(
            video_path,
            [Path(path) for path in dynamic_tail_result["frame_paths"]],
        )

    return {
        "npz_path": str(npz_path),
        "pred_key": pred_key,
        "alignment_mode": pred_pairs.alignment_mode,
        "projected_samples": len(pred_pairs.indexed_samples),
        "direct_image_space_scanpath": str(direct_path),
        "direct_image_space_points": direct_count,
        "reference_frame_scanpath_overlay": str(
            pred_dir / "pred_reference_frame_scanpath_overlay.png"
        ),
        "reference_frame_scanpath_clean": str(pred_dir / "pred_reference_frame_scanpath_clean.png"),
        "dynamic_tail": dynamic_tail_result,
        "dynamic_tail_video": (
            str(video_path)
            if write_video and dynamic_tail_result and dynamic_tail_result["frame_paths"]
            else None
        ),
    }


def write_hagi_scanpath_outputs(
    *,
    gt_provider: Any | None,
    output_dir: Path,
    hagi_npz: Path,
    hagi_pairs: PredictionPairs,
    stream_id_value: str,
    make_upright: bool,
    write_video: bool,
    tail_frames: int,
) -> dict[str, Any]:
    """Write image-space scanpath outputs for aligned HAGI++ samples."""

    if gt_provider is None:
        raise ValueError("HAGI++ scanpath projection requires an ADT provider.")
    if not hagi_pairs.indexed_samples:
        raise ValueError("No HAGI++ samples could be aligned to the selected GT window.")

    hagi_dir = output_dir / "hagi"
    hagi_dir.mkdir(parents=True, exist_ok=True)
    hagi_samples = [sample for _, sample in hagi_pairs.indexed_samples]
    hagi_orders = [index for index, _ in hagi_pairs.indexed_samples]

    direct_path = hagi_dir / "hagi_image_space_scanpath_overlay.png"
    direct_count = write_direct_image_space_scanpath(
        direct_path,
        gt_provider=gt_provider,
        indexed_samples=hagi_pairs.indexed_samples,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
        title="HAGI++ image-space scanpath",
    )

    reference_scanpath = reference_scanpath_from_samples(
        gt_provider,
        hagi_samples,
        hagi_orders,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    write_reference_frame_scanpath_overlay(
        hagi_dir / "hagi_reference_frame_scanpath_overlay.png",
        reference_scanpath,
    )
    write_reference_frame_scanpath_clean(
        hagi_dir / "hagi_reference_frame_scanpath_clean.png",
        reference_scanpath,
    )

    dynamic_tail_result = write_dynamic_tail_overlay_frames(
        gt_provider,
        hagi_pairs.indexed_samples,
        hagi_dir / "hagi_dynamic_tail_overlays",
        title_prefix="HAGI++ dynamic scanpath tail",
        tail_frames=tail_frames,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    video_path = hagi_dir / "hagi_dynamic_tail_video.mp4"
    if write_video:
        write_video_from_overlay_frames(
            video_path,
            [Path(path) for path in dynamic_tail_result["frame_paths"]],
        )

    return {
        "npz_path": str(hagi_npz),
        "alignment_mode": hagi_pairs.alignment_mode,
        "projected_samples": len(hagi_pairs.indexed_samples),
        "direct_image_space_scanpath": str(direct_path),
        "direct_image_space_points": direct_count,
        "reference_frame_scanpath_overlay": str(
            hagi_dir / "hagi_reference_frame_scanpath_overlay.png"
        ),
        "reference_frame_scanpath_clean": str(hagi_dir / "hagi_reference_frame_scanpath_clean.png"),
        "dynamic_tail": dynamic_tail_result,
        "dynamic_tail_video": (
            str(video_path)
            if write_video and dynamic_tail_result and dynamic_tail_result["frame_paths"]
            else None
        ),
    }


def write_comparison_dynamic_tail_outputs(
    *,
    gt_provider: Any,
    output_dir: Path,
    reference_pairs: list[tuple[int, GazeSample]],
    pred_pairs: PredictionPairs | None,
    hagi_pairs: PredictionPairs | None,
    tail_frames: int,
    stream_id_value: str,
    make_upright: bool,
    write_video: bool,
) -> dict[str, Any]:
    """Write one dynamic-tail video with GT, SparseGaze, and HAGI++ tracks."""

    tracks: dict[str, list[tuple[int, GazeSample]]] = {"GT": reference_pairs}
    if pred_pairs is not None and pred_pairs.indexed_samples:
        tracks["SparseGaze"] = pred_pairs.indexed_samples
    if hagi_pairs is not None and hagi_pairs.indexed_samples:
        tracks["HAGI++"] = hagi_pairs.indexed_samples

    comparison_dir = output_dir / "comparison"
    result = write_multi_track_dynamic_tail_overlay_frames(
        gt_provider,
        reference_pairs=reference_pairs,
        tracks=tracks,
        output_dir=comparison_dir / "dynamic_tail_overlays",
        tail_frames=tail_frames,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    video_path = comparison_dir / "gt_sparsegaze_hagi_dynamic_tail_video.mp4"
    if write_video:
        write_video_from_overlay_frames(
            video_path,
            [Path(path) for path in result["frame_paths"]],
        )
    return {
        "tracks": list(tracks),
        "dynamic_tail": result,
        "dynamic_tail_video": (
            str(video_path) if write_video and result and result["frame_paths"] else None
        ),
    }


class PredictionPairs:
    def __init__(self, indexed_samples: list[tuple[int, GazeSample]], alignment_mode: str) -> None:
        self.indexed_samples = indexed_samples
        self.alignment_mode = alignment_mode


def prediction_pairs_from_npz(
    *,
    gt_provider: Any,
    npz_path: Path,
    pred_key: str,
    gt_pairs: list[tuple[int, GazeSample]],
    all_gt_samples: list[GazeSample],
    depth_mode: str,
    fixed_depth_m: float,
    stream_id_value: str,
    make_upright: bool,
) -> PredictionPairs:
    """Align prediction directions to GT frames and project them to RGB pixels."""

    prediction = load_prediction_arrays(npz_path, pred_key)
    directions = prediction["directions"]
    timestamps = prediction.get("timestamps_ns")

    if timestamps is not None:
        direction_by_timestamp = {
            int(timestamp_ns): directions[index] for index, timestamp_ns in enumerate(timestamps)
        }
        aligned = [
            (gt_index, gt_sample, direction_by_timestamp.get(gt_sample.query_timestamp_ns))
            for gt_index, gt_sample in gt_pairs
        ]
        alignment_mode = "timestamp"
    else:
        aligned = [
            (
                gt_index,
                gt_sample,
                directions[gt_index] if 0 <= gt_index < len(directions) else None,
            )
            for gt_index, gt_sample in gt_pairs
        ]
        alignment_mode = "frame_index"

    indexed_samples: list[tuple[int, GazeSample]] = []
    for gt_index, gt_sample, direction in aligned:
        if direction is None:
            continue
        pred_sample = prediction_sample_from_direction(
            gt_provider=gt_provider,
            gt_sample=gt_sample,
            direction=direction,
            depth_mode=depth_mode,
            fixed_depth_m=fixed_depth_m,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
        )
        indexed_samples.append((gt_index, pred_sample))

    if timestamps is None and len(directions) != len(all_gt_samples):
        alignment_mode = f"frame_index_len_pred_{len(directions)}_gt_{len(all_gt_samples)}"
    return PredictionPairs(indexed_samples, alignment_mode)


def hagi_pairs_from_npz(
    *,
    gt_provider: Any,
    hagi_npz: Path,
    hagi_adt_data: Path,
    sequence_name: str,
    gt_pairs: list[tuple[int, GazeSample]],
    all_gt_samples: list[GazeSample],
    depth_mode: str,
    fixed_depth_m: float,
    stream_id_value: str,
    make_upright: bool,
) -> PredictionPairs:
    """Align HAGI++ CPF predictions to GT frames and project them to RGB pixels."""

    frame_directions = load_hagi_world_directions(
        hagi_npz=hagi_npz,
        hagi_adt_data=hagi_adt_data,
        sequence_name=sequence_name,
    )
    indexed_samples: list[tuple[int, GazeSample]] = []
    for gt_index, gt_sample in gt_pairs:
        direction = frame_directions.get(gt_index)
        if direction is None:
            continue
        if gt_index >= len(all_gt_samples):
            continue
        hagi_sample = prediction_sample_from_direction(
            gt_provider=gt_provider,
            gt_sample=gt_sample,
            direction=direction,
            depth_mode=depth_mode,
            fixed_depth_m=fixed_depth_m,
            stream_id_value=stream_id_value,
            make_upright=make_upright,
        )
        indexed_samples.append((gt_index, replace(
            hagi_sample,
            validation_notes=f"hagi++;depth_mode={depth_mode}",
        )))
    return PredictionPairs(indexed_samples, "frame_index")


def load_hagi_world_directions(
    *,
    hagi_npz: Path,
    hagi_adt_data: Path,
    sequence_name: str,
) -> dict[int, np.ndarray]:
    """Return HAGI++ directions in ADT world/Scene frame by frame index."""

    adt_data = np.load(hagi_adt_data, allow_pickle=True).item()
    if sequence_name not in adt_data:
        raise KeyError(f"{sequence_name} not found in HAGI ADT cache: {hagi_adt_data}")
    record = adt_data[sequence_name]
    transforms = np.asarray(record["T_world_CPF"], dtype=np.float64)

    with np.load(hagi_npz, allow_pickle=True) as data:
        sequence_mask = data["sequence_name"].astype(str) == sequence_name
        frames = np.asarray(data["frame_index"][sequence_mask], dtype=np.int64)
        pred_cpf = normalize_vectors(np.asarray(data["pred"][sequence_mask], dtype=np.float64))

    valid = (frames >= 0) & (frames < len(transforms))
    frames = frames[valid]
    pred_cpf = pred_cpf[valid]
    rotations = transforms[frames, :3, :3]
    pred_world = normalize_vectors(np.einsum("fij,fj->fi", rotations, pred_cpf))
    return {int(frame): pred_world[index] for index, frame in enumerate(frames)}


def load_prediction_arrays(npz_path: Path, pred_key: str) -> dict[str, Any]:
    """Load prediction directions plus optional timestamps from one NPZ."""

    with np.load(npz_path, allow_pickle=True) as data:
        if pred_key not in data.files:
            raise KeyError(f"{npz_path} does not contain prediction key {pred_key!r}")
        directions = normalize_vectors(np.asarray(data[pred_key], dtype=np.float64))
        if directions.ndim != 2 or directions.shape[1] != 3:
            raise ValueError(f"{pred_key} must have shape (N, 3), got {directions.shape}")
        result: dict[str, Any] = {"directions": directions}
        if "timestamps_ns" in data.files:
            timestamps = np.asarray(data["timestamps_ns"], dtype=np.int64).reshape(-1)
            if len(timestamps) != len(directions):
                raise ValueError("timestamps_ns length does not match prediction directions")
            result["timestamps_ns"] = timestamps
        return result


def prediction_sample_from_direction(
    *,
    gt_provider: Any,
    gt_sample: GazeSample,
    direction: np.ndarray,
    depth_mode: str,
    fixed_depth_m: float,
    stream_id_value: str,
    make_upright: bool,
) -> GazeSample:
    """Create a GazeSample-like prediction row from one aligned GT frame."""

    origin = sample_origin_scene(gt_sample)
    if origin is None:
        return replace(
            gt_sample,
            gaze_valid=False,
            projection_valid=False,
            gaze_u_px=None,
            gaze_v_px=None,
            projection_in_image=False,
            validation_notes="prediction_origin_unavailable",
        )

    depth = resolve_prediction_depth(gt_sample, depth_mode, fixed_depth_m)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    point = origin + direction * depth
    projection, image_size = project_scene_points_to_rgb(
        gt_provider,
        [point],
        gt_sample.query_timestamp_ns,
        stream_id_value=stream_id_value,
        make_upright=make_upright,
    )
    width, height = image_size
    uv = projection[0]
    projection_valid = uv is not None
    gaze_u = float(uv[0]) if projection_valid else None
    gaze_v = float(uv[1]) if projection_valid else None
    in_image = bool(
        projection_valid
        and gaze_u is not None
        and gaze_v is not None
        and 0 <= gaze_u < width
        and 0 <= gaze_v < height
    )

    return replace(
        gt_sample,
        depth_m=float(depth),
        projection_valid=projection_valid,
        gaze_u_px=gaze_u,
        gaze_v_px=gaze_v,
        projection_in_image=in_image,
        image_width_px=width,
        image_height_px=height,
        gaze_point_scene_x_m=float(point[0]),
        gaze_point_scene_y_m=float(point[1]),
        gaze_point_scene_z_m=float(point[2]),
        gaze_dir_scene_unit_x=float(direction[0]),
        gaze_dir_scene_unit_y=float(direction[1]),
        gaze_dir_scene_unit_z=float(direction[2]),
        validation_notes=f"sparsegaze_prediction;depth_mode={depth_mode}",
    )


def write_dynamic_tail_overlay_frames(
    gt_provider: Any,
    indexed_samples: list[tuple[int, GazeSample]],
    output_dir: Path,
    *,
    title_prefix: str,
    tail_frames: int,
    stream_id_value: str,
    make_upright: bool,
) -> dict[str, Any]:
    """Render per-frame overlays with past gaze points reprojected to the current frame."""

    if tail_frames < 0:
        raise ValueError("tail_frames must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    tail_point_counts: list[int] = []
    skipped_frames = 0
    for current_position, (current_index, current_sample) in enumerate(indexed_samples):
        image_with_dt = get_rgb_image(
            gt_provider,
            current_sample.query_timestamp_ns,
            stream_id_value=stream_id_value,
        )
        if not image_with_dt.is_valid():
            skipped_frames += 1
            continue

        tail_pairs = [
            (row_index, sample)
            for row_index, sample in indexed_samples[: current_position + 1]
            if current_index - tail_frames <= row_index <= current_index
        ]
        scene_points = [sample_scene_point(sample) for _, sample in tail_pairs]
        try:
            projections, image_size = project_scene_points_to_rgb(
                gt_provider,
                scene_points,
                current_sample.query_timestamp_ns,
                stream_id_value=stream_id_value,
                make_upright=make_upright,
            )
        except ValueError:
            skipped_frames += 1
            continue

        image = image_with_dt.data().to_numpy_array()
        if make_upright:
            image = np.rot90(image, k=3)

        projected_points = filter_in_image_points(tail_pairs, projections, image_size)
        tail_point_counts.append(len(projected_points))
        output_path = (
            output_dir / f"tail_row{current_index:04d}_{current_sample.query_timestamp_ns}.png"
        )
        fig = render_dynamic_tail_figure(
            image=image,
            projected_points=projected_points,
            current_index=current_index,
            title=(
                f"{title_prefix} | frame={current_index} | "
                f"tail={tail_frames} frames | in_image={len(projected_points)}/{len(tail_pairs)}"
            ),
        )
        fig.savefig(output_path, dpi=140)
        pyplot().close(fig)
        output_paths.append(output_path)

    return {
        "frame_dir": str(output_dir),
        "frame_paths": [str(path) for path in output_paths],
        "frames_rendered": len(output_paths),
        "frames_requested": len(indexed_samples),
        "frames_skipped": skipped_frames,
        "tail_frames": tail_frames,
        "mean_visible_tail_points": float(np.mean(tail_point_counts)) if tail_point_counts else 0.0,
    }


def write_multi_track_dynamic_tail_overlay_frames(
    gt_provider: Any,
    *,
    reference_pairs: list[tuple[int, GazeSample]],
    tracks: dict[str, list[tuple[int, GazeSample]]],
    output_dir: Path,
    tail_frames: int,
    stream_id_value: str,
    make_upright: bool,
) -> dict[str, Any]:
    """Render current-frame overlays with multiple method tails."""

    if tail_frames < 0:
        raise ValueError("tail_frames must be non-negative")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths: list[Path] = []
    skipped_frames = 0
    visible_counts: dict[str, list[int]] = {name: [] for name in tracks}
    for current_index, current_sample in reference_pairs:
        image_with_dt = get_rgb_image(
            gt_provider,
            current_sample.query_timestamp_ns,
            stream_id_value=stream_id_value,
        )
        if not image_with_dt.is_valid():
            skipped_frames += 1
            continue

        image = image_with_dt.data().to_numpy_array()
        if make_upright:
            image = np.rot90(image, k=3)

        projected_tracks: dict[str, list[tuple[int, float, float]]] = {}
        for name, pairs in tracks.items():
            tail_pairs = [
                (row_index, sample)
                for row_index, sample in pairs
                if current_index - tail_frames <= row_index <= current_index
            ]
            scene_points = [sample_scene_point(sample) for _, sample in tail_pairs]
            try:
                projections, image_size = project_scene_points_to_rgb(
                    gt_provider,
                    scene_points,
                    current_sample.query_timestamp_ns,
                    stream_id_value=stream_id_value,
                    make_upright=make_upright,
                )
            except ValueError:
                projected_tracks[name] = []
                visible_counts[name].append(0)
                continue
            points = filter_in_image_points(tail_pairs, projections, image_size)
            projected_tracks[name] = points
            visible_counts[name].append(len(points))

        filename = f"comparison_row{current_index:04d}_{current_sample.query_timestamp_ns}.png"
        output_path = output_dir / filename
        fig = render_multi_track_tail_figure(
            image=image,
            projected_tracks=projected_tracks,
            current_index=current_index,
            title=(
                f"GT vs SparseGaze vs HAGI++ | frame={current_index} | "
                f"tail={tail_frames} frames"
            ),
        )
        fig.savefig(output_path, dpi=140)
        pyplot().close(fig)
        output_paths.append(output_path)

    return {
        "frame_dir": str(output_dir),
        "frame_paths": [str(path) for path in output_paths],
        "frames_rendered": len(output_paths),
        "frames_requested": len(reference_pairs),
        "frames_skipped": skipped_frames,
        "tail_frames": tail_frames,
        "mean_visible_tail_points": {
            name: float(np.mean(counts)) if counts else 0.0
            for name, counts in visible_counts.items()
        },
    }


def render_dynamic_tail_figure(
    *,
    image: np.ndarray,
    projected_points: list[tuple[int, float, float]],
    current_index: int,
    title: str,
) -> Any:
    """Render one current-frame image with the visible dynamic scanpath tail."""

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_axis_off()
    if projected_points:
        orders = [row_index for row_index, _, _ in projected_points]
        xs = [u_px for _, u_px, _ in projected_points]
        ys = [v_px for _, _, v_px in projected_points]
        ax.plot(xs, ys, color="white", linewidth=2.2, alpha=0.82)
        ax.plot(xs, ys, color="black", linewidth=1.1, alpha=0.9)
        ax.scatter(
            xs,
            ys,
            c=orders,
            cmap="viridis",
            s=42,
            edgecolors="white",
            linewidths=0.55,
        )
        ax.scatter([xs[0]], [ys[0]], marker="o", color="lime", s=80, edgecolors="black")
        current_positions = [
            (u_px, v_px)
            for row_index, u_px, v_px in projected_points
            if row_index == current_index
        ]
        if current_positions:
            current_u, current_v = current_positions[-1]
            ax.scatter(
                [current_u],
                [current_v],
                marker="X",
                color="yellow",
                s=120,
                edgecolors="black",
                linewidths=0.7,
            )
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def render_multi_track_tail_figure(
    *,
    image: np.ndarray,
    projected_tracks: dict[str, list[tuple[int, float, float]]],
    current_index: int,
    title: str,
) -> Any:
    """Render one current-frame image with GT/SparseGaze/HAGI++ tails."""

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.set_axis_off()
    for name, points in projected_tracks.items():
        if not points:
            continue
        color = TRACK_COLORS.get(name, "#4c78a8")
        xs = [u_px for _, u_px, _ in points]
        ys = [v_px for _, _, v_px in points]
        ax.plot(xs, ys, color="white", linewidth=3.0, alpha=0.72)
        ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.95, label=name)
        ax.scatter(
            xs,
            ys,
            s=34,
            color=color,
            edgecolors="white",
            linewidths=0.55,
            alpha=0.95,
        )
        current_positions = [
            (u_px, v_px)
            for row_index, u_px, v_px in points
            if row_index == current_index
        ]
        if current_positions:
            current_u, current_v = current_positions[-1]
            ax.scatter(
                [current_u],
                [current_v],
                marker="X",
                color=color,
                s=115,
                edgecolors="black",
                linewidths=0.75,
            )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower right", frameon=True, fontsize=8)
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    return fig


def filter_in_image_points(
    tail_pairs: list[tuple[int, GazeSample]],
    projections: list[np.ndarray | None],
    image_size: tuple[int, int],
) -> list[tuple[int, float, float]]:
    """Keep projected tail points that are visible in the current RGB frame."""

    width, height = image_size
    points: list[tuple[int, float, float]] = []
    for (row_index, _), projection in zip(tail_pairs, projections, strict=True):
        if projection is None:
            continue
        u_px = float(projection[0])
        v_px = float(projection[1])
        if 0 <= u_px < width and 0 <= v_px < height:
            points.append((row_index, u_px, v_px))
    return points


def write_direct_image_space_scanpath(
    path: Path,
    *,
    gt_provider: Any | None,
    indexed_samples: list[tuple[int, GazeSample]],
    stream_id_value: str,
    make_upright: bool,
    title: str,
) -> int:
    """Draw per-frame image-space gaze coordinates over the final RGB frame."""

    if not indexed_samples:
        raise ValueError("No samples available for scanpath drawing")

    reference_sample = indexed_samples[-1][1]
    if gt_provider is None:
        image = blank_image_from_samples([sample for _, sample in indexed_samples])
    else:
        image_with_dt = get_rgb_image(
            gt_provider,
            reference_sample.query_timestamp_ns,
            stream_id_value=stream_id_value,
        )
        if not image_with_dt.is_valid():
            raise ValueError(
                f"Reference RGB image is invalid at {reference_sample.query_timestamp_ns}"
            )

        image = image_with_dt.data().to_numpy_array()
        if make_upright:
            image = np.rot90(image, k=3)

    points = [
        (row_index, sample.gaze_u_px, sample.gaze_v_px)
        for row_index, sample in indexed_samples
        if sample.projection_valid
        and sample.projection_in_image
        and sample.gaze_u_px is not None
        and sample.gaze_v_px is not None
    ]
    if not points:
        raise ValueError("No in-image gaze points available for direct image-space scanpath")

    orders = [row_index for row_index, _, _ in points]
    xs = [float(u_px) for _, u_px, _ in points]
    ys = [float(v_px) for _, _, v_px in points]

    plt = pyplot()
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(image)
    ax.plot(xs, ys, color="white", linewidth=2.0, alpha=0.8)
    ax.plot(xs, ys, color="black", linewidth=1.0, alpha=0.85)
    scatter = ax.scatter(
        xs,
        ys,
        c=orders,
        cmap="viridis",
        s=46,
        edgecolors="white",
        linewidths=0.6,
    )
    ax.scatter([xs[0]], [ys[0]], marker="o", color="lime", s=90, edgecolors="black")
    ax.scatter([xs[-1]], [ys[-1]], marker="X", color="yellow", s=110, edgecolors="black")
    if len(xs) <= 25:
        for order, u_px, v_px in zip(orders, xs, ys, strict=True):
            ax.text(
                u_px + 4,
                v_px + 4,
                str(order),
                color="white",
                fontsize=7,
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
            )
    ax.set_axis_off()
    ax.set_title(
        f"{title} (ref frame={orders[-1]}, in_image={len(xs)}/{len(indexed_samples)})",
        fontsize=9,
    )
    fig.colorbar(scatter, ax=ax, label="GT row/frame index")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return len(xs)


def sample_scene_point(sample: GazeSample) -> np.ndarray:
    values = [
        sample.gaze_point_scene_x_m,
        sample.gaze_point_scene_y_m,
        sample.gaze_point_scene_z_m,
    ]
    if any(value is None for value in values):
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    point = np.asarray(values, dtype=np.float64)
    if not np.isfinite(point).all():
        return np.array([np.nan, np.nan, np.nan], dtype=np.float64)
    return point


def blank_image_from_samples(samples: list[GazeSample]) -> np.ndarray:
    """Create a neutral image-space canvas when the ADT provider is unavailable."""

    for sample in samples:
        if sample.image_width_px is not None and sample.image_height_px is not None:
            width = int(sample.image_width_px)
            height = int(sample.image_height_px)
            return np.full((height, width, 3), 245, dtype=np.uint8)
    raise ValueError("Cannot infer image size from selected samples")


def sample_origin_scene(sample: GazeSample) -> np.ndarray | None:
    values = [
        sample.gaze_origin_scene_x_m,
        sample.gaze_origin_scene_y_m,
        sample.gaze_origin_scene_z_m,
    ]
    if any(value is None for value in values):
        return None
    origin = np.asarray(values, dtype=np.float64)
    return origin if np.isfinite(origin).all() else None


def resolve_prediction_depth(sample: GazeSample, depth_mode: str, fixed_depth_m: float) -> float:
    if depth_mode == "fixed":
        return float(fixed_depth_m)
    if depth_mode == "gt_depth":
        if sample.depth_m is not None and np.isfinite(sample.depth_m) and sample.depth_m > 0:
            return float(sample.depth_m)
        return float(fixed_depth_m)
    raise ValueError(f"Unsupported depth_mode: {depth_mode}")


def resolve_reports_dir(explicit_reports_dir: Path | None) -> Path:
    if explicit_reports_dir is not None:
        return reports_root(explicit_reports_dir)
    env_reports_dir = os.environ.get("REPORTS_DIR")
    if env_reports_dir:
        return reports_root(env_reports_dir)
    structured_reports_dir = Path("/mnt/d/SparseGaze/ADT-Gaze-structured")
    if structured_reports_dir.exists():
        return structured_reports_dir
    return reports_root(REPO_ROOT / "outputs" / "reports")


def find_gaze_samples_csv(reports_dir: Path, sequence_name: str) -> Path:
    try:
        return find_sequence_file(reports_dir, sequence_name, "gaze", "gaze_samples.csv")
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            "Could not find gaze_samples.csv. Pass --input-csv or --reports-dir.\n"
            f"Current reports_dir: {reports_dir}\n"
            f"Original error: {exc}"
        ) from exc


def default_run_name(start_frame: int, end_frame: int | None, stride: int) -> str:
    end_label = "end" if end_frame is None else str(end_frame)
    return f"frames_{start_frame}_{end_label}_stride_{stride}"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def pyplot() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    return plt


if __name__ == "__main__":
    main()
