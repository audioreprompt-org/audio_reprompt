from typing import Dict
from metrics.core.base import Metric
from metrics.implementation.clap_metric import CLAPMetric
# from metrics.impl.fad_metric import FADMetric   # TODO
# from metrics.impl.aesthetics_metric import AestheticsMetric  # TODO

def build_metric_registry() -> Dict[str, Metric]:
    return {
        "clap": CLAPMetric(),
    }
