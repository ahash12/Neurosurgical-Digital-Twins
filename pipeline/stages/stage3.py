import numpy as np

from pipeline.scripts.stage3_postlateral import (
    posterolateral_score_from_ijk,
    resolve_mri_thresholds,
    resolve_thresholds,
)
from utils import VERTEBRA_LABELS

from pipeline.stages.base import PipelineStage, register_stage


@register_stage("stage3")
class Stage3Posterolateral(PipelineStage):
    def prepare(self, context):
        if getattr(context, "modality", "CT") == "MR":
            return resolve_mri_thresholds(
                context.ct,
                context.seg,
                getattr(context, "sequence", ""),
            )
        low_hu, high_hu, threshold_meta = resolve_thresholds(context.ct, context.seg)
        return low_hu, high_hu, threshold_meta

    def run_vertebra(self, context, vertebra, thresholds=None):
        label = VERTEBRA_LABELS[str(vertebra).upper()]
        low_signal, high_signal, threshold_meta = thresholds or self.prepare(context)
        ijk = np.argwhere(context.seg == label)
        signal = threshold_meta.get("feature_image", context.ct)
        return posterolateral_score_from_ijk(
            signal,
            context.affine,
            ijk,
            lesion_low_hu=low_signal,
            lesion_high_hu=high_signal,
            threshold_meta=threshold_meta,
        )
