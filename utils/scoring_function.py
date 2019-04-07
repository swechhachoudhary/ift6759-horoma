import numpy as np
from sklearn.metrics import recall_score, f1_score

# Note : this code comes from OM Signal block 2 baseline
# Some changes were made to be able to use this for Horoma


def scorePerformance(treeClass_pred, treeClass_true):
    """
    Computes the combined multitask performance score. The 3 regression tasks are individually scored using Kendalls
    correlation coeffficient. the user classification task is scored according to macro averaged recall,
    with an adjustment for chance level. All performances are clipped at 0.0, so that zero indicates chance
    or worse performance, and 1.0 indicates perfect performance. The individual performances are then combined by taking
    the geometric mean.
    :param treeClass_pred: 1D int32 numpy array. containing the predicted tree class label
    :param treeClass_true: 1D int32 numpy array. containing the true tree class label
    :return: The combined performance score on all tasks; 0.0 means at least one task has chance level performance or worse, 1.0 means all tasks are solved perfectly.
    The individual task performance scores are also returned
    """

    numElmts = None

    # Input checking
    if treeClass_true is not None:
        assert isinstance(treeClass_pred, np.ndarray)
        assert len(treeClass_pred.shape) == 1
        assert treeClass_pred.dtype == np.int32

        assert isinstance(treeClass_true, np.ndarray)
        assert len(treeClass_true.shape) == 1
        assert treeClass_true.dtype == np.int32

        assert (len(treeClass_pred) == len(treeClass_true))
        if numElmts is not None:
            assert (len(treeClass_pred) == numElmts) and (
                len(treeClass_true) == numElmts)
        else:
            numElmts = len(treeClass_pred)

    if numElmts is None:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    numVal = 0.0

    # Accuracy is computed with macro averaged recall so that accuracy is computed as though the classes
    # were balanced, even if they are not. Note that the training, validation and testing sets are balanced as given.
    # Unbalanced classes would only be and issue if a new train/validation split is created.
    # Any accuracy value worse than random chance will be clipped at zero.
    if treeClass_true is not None:
        numVal += 1.0
        treeClassAccuracy = recall_score(treeClass_true, treeClass_pred, average='macro')
        adjustementTerm = 1.0 / len(np.unique(treeClass_true))
        treeClassAccuracy = (treeClassAccuracy - adjustementTerm) / \
            (1 - adjustementTerm)
        if treeClassAccuracy < 0 or np.isnan(treeClassAccuracy):
            treeClassAccuracy = 0.0
        treeClassAccuracyRep = treeClassAccuracy
        treeClassF1 = f1_score(treeClass_true, treeClass_pred, average="weighted")

    else:
        treeClassAccuracy = 1.0
        treeClassAccuracyRep = 0.0

    # Compute the final performance score as the geometric mean of the individual task performances.
    # A high geometric mean ensures that there are no tasks with very poor performance that are masked by good
    # performance on the other tasks. If any task has chance performance or worse,
    # the overall performance will be zero. If all tasks are perfectly solved,
    # the overall performance will be 1.
    combinedPerformanceScore = np.power(
        treeClassAccuracy,
        1.0 / max(1.0, numVal)
    )

    return (
        combinedPerformanceScore,
        treeClassAccuracyRep,
        treeClassF1
    )
