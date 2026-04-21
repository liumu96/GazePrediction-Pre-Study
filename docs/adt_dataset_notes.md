# ADT Dataset Notes

ADT sequences commonly combine:

- VRS streams: egocentric video, depth images, and segmentations.
- CSV annotations: 2D boxes, 3D boxes, scene objects, trajectories, and gaze.
- JSON metadata: sequence metadata, instance metadata, skeleton transforms, and
  MPS summaries.
- MPS outputs: eye gaze and other machine perception service products when
  available.

Useful references:

- Official Project Aria ADT dataset documentation.
- `projectaria_tools.projects.adt` in the installed `projectaria-tools` package.
- Optional local source checkout under `external/projectaria_tools/` if you keep
  one for reading examples and implementation details.

Keep notes here when a dataset field, coordinate convention, or timestamp
alignment rule becomes clear from an experiment.
