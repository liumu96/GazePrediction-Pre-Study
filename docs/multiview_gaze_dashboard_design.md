# Multi-View Gaze Dashboard Design Notes

This document explains the purpose and visual design of
`Experiments/visualization & Analysis/ADT/notebooks/07_multiview_gaze_dashboard.ipynb`.

The notebook is an interactive figure finder, not the final paper figure
itself. Its role is to help select a sequence, frame window, and focus frame.
After a useful case is found, the paper should use a static export or screenshot
of that selected view with a concise caption.

## Figure Goal

The dashboard is designed to answer one practical question:

> Given a short ADT window, how do local gaze motion, image-space gaze,
> scene-level event labels, object hits, body pose, and 3D scene geometry relate
> to each other?

It is not meant to show every extracted feature. Features that do not help this
question should stay out of the dashboard, otherwise the figure becomes a debug
dump rather than visual evidence.

## Panels

### A. Local eye-in-head gaze

This panel plots CPF-local yaw and pitch in degrees. CPF-local gaze is useful
for seeing eye-in-head motion, independent of how the head moves through the
Scene frame. The shaded background comes from Scene-direction event labels:
green means fixation and orange means transition.

Use this panel to check whether a selected transition corresponds to a clear
local gaze change or whether the event is mainly caused by head/scene motion.

### B. Motion magnitude and event context

This panel compares CPF-local gaze velocity, Scene gaze angular velocity, and
head rotation speed. It is a magnitude view, not a direction view. It helps
answer whether a visually important interval is dominated by eye motion, head
motion, or both.

Use this panel before making claims such as "head motion explains this gaze
change." If head rotation is low while Scene gaze velocity is high, the segment
is more likely eye-driven. If both are high, the segment is a stronger candidate
for head-gaze coupling analysis.

### C. RGB image-space gaze

This panel shows the scanpath in RGB pixel coordinates using the actual image
width and height. The y-axis is inverted to match image coordinates. The panel
does not display the RGB frame itself; it shows where the projected gaze lies
within the image plane.

Use this panel to verify whether a selected 3D gaze/ray behavior also looks
reasonable in the camera view.

### D. Object-hit context

This panel shows the ray-box first-hit distance and the gap between ADT's
depth-defined gaze point and the ray-box hit point. It also displays the same
Scene event labels as a compact timeline.

Use this panel to check whether the selected focus frame has a meaningful object
hit and whether the object-box approximation agrees with the depth-defined gaze
point. A large depth-hit gap is a warning that the cuboid hit may not represent
the true physical surface point.

### E. 3D scene context

This panel is the main spatial view. It shows static and dynamic object cuboids,
skeleton pose, head/device trajectory, gaze rays, depth-defined gaze points,
and the current ray-box hit when available.

Use this panel to choose a paper snapshot. The snapshot should focus on one
clear story, for example:

- a fixation interval where the gaze ray repeatedly hits the same object box;
- a transition interval where gaze rays sweep across objects;
- a frame where depth-defined gaze point and ray-box hit disagree.

## Global Color Policy

The dashboard uses a shared semantic color palette instead of local per-plot
color choices. This matters because the same visual meaning should keep the
same color across the interactive notebooks, static figures, and future paper
plots. The current implementation uses the Okabe-Ito palette as the base
because it is common in scientific figures and is more colorblind-safe than
arbitrary saturated colors.

The implementation lives in:

- `visualization/viz_palette.py`

Viewer modules should import from this file rather than defining their own
colors. This prevents the same object, event, or gaze signal from changing
colors across notebooks.

| Semantic item | Color | Meaning |
| --- | --- | --- |
| GT / first gaze track | blue `#0072B2` | primary gaze signal |
| Prediction tracks | orange/green/purple variants | future model outputs |
| Gaze rays / gaze points | vermillion `#D55E00` | gaze geometry in 3D |
| Head trajectory / head speed | near-black `#222222` | head/device motion |
| Skeleton | bluish green `#009E73` | body pose context |
| Fixation background | transparent green | Scene-direction fixation |
| Transition background | transparent orange | non-fixation valid frames |
| Ray-box hit point / outline | blue `#0072B2` | first object-box intersection |
| Static boxes | neutral gray | scene structure |
| Dynamic boxes | orange | moving or timestamped objects |

The palette should be kept consistent with future SparseGaze paper figures. If
the main paper adopts a different global palette, change the constants in:

- `visualization/viz_palette.py`

## Interaction and Paper Export

The interactive controls are for exploration:

- sequence: select the ADT sequence;
- start/end: select the frame window;
- stride: reduce visual density;
- focus: select the frame highlighted by the vertical line and current 3D hit;
- category/exclude: include or remove object categories;
- ray scale: fixed-length ray for direction comparison or depth-scaled ray for
  endpoint context.

For paper use, do not include the widget controls. Select a stable view, hide
browser/notebook chrome, and export or screenshot the figure area only. The
caption should state the selected sequence, frame window, focus frame, and what
the reader should notice.

## Limitations

The dashboard still uses object cuboids, not meshes. A ray-box hit is therefore
a coarse semantic approximation. The highlighted blue outline means the current
gaze ray intersects that object cuboid; it does not mean confirmed attention or
fixation on that object. The object box should still be present in the normal
static/dynamic object layer, and the blue outline is only an additional
frame-level style cue.

The depth-defined gaze point comes from ADT gaze depth and can differ from the
box intersection. This difference is not a bug; it is one of the diagnostics
shown in panel D.
