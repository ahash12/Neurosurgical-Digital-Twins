"""Moved: MRI geometry now lives with the step that uses it.

    vertebral body extraction / spine axes   sins_steps.common.body
    collapse heights                         sins_steps.step4_collapse.collapse
    canal centre, robust z thresholds        sins_steps.step3_posterolateral.posterolateral
    vertebral centres                        sins_steps.common.case.sagittal_centroids
"""
from sins_steps.common.body import body_centroid, body_points, mid_slices, spine_axes  # noqa: F401
from sins_steps.common.case import world  # noqa: F401
from sins_steps.step3_posterolateral.posterolateral import canal_center, erode_in_plane, robust_z_thresholds  # noqa: F401
