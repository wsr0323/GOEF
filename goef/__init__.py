"""GOEF: Globality-Orientated Evaluation Framework for few-shot image anomaly detection.

The package implements the evaluation framework proposed in

    "Don't Miss Forest for A Tree: The Globality-Orientated Evaluation
     Framework for Few-shot Image Anomaly Detection"

and contains the reference implementation of every stage used in the paper:

``goef.features``    DINOv3 image embeddings used for representative selection.
``goef.selection``   The five reference-selection strategies compared in Sec. III-D.
``goef.centers``     Materialisation of the k-shot "center" datasets.
``goef.weights``     The globality weights of Eq. (21)-(22) for arbitrary k.
``goef.aggregation`` Globality-preserving metric aggregation (Algo. 2).
``goef.validation``  Approximation error eps, ranks r, and Table I.
``goef.reevaluate``  Weighted k-shot re-evaluation and the paper's rankings.

The four image/pixel metrics shared by every evaluated method live in
``modifications/common/unified_evaluation.py``: that file is copied unchanged into
each third-party project, so it deliberately has no dependency on this package.
"""

__version__ = "1.1.0"

__all__ = [
    "aggregation",
    "centers",
    "features",
    "sampling",
    "selection",
    "validation",
    "weights",
]
