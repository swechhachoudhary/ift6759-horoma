import torch
from tqdm import tqdm
import numpy as np
import utils.scoring_function as scoreF

# Note : this code comes from OM Signal block 2 baseline
# Some modifications were made to make it work with Horoma
# (For example, evaluation is made on the class of the tree now)


def get_scoring_func_param_index(target_labels):
    scoring_func_param_index = [
        None if target_labels.count(
            'tree_class') == 0 else target_labels.index('tree_class'),
    ]
    return scoring_func_param_index


def update_prediction_data(score_index, y, outputs, TC):
    score_param_index = score_index

    treeClass_true, treeClass_pred = TC

    if score_param_index[0] is not None:
        i = score_param_index[0]
        _, pred_classes = torch.max(outputs[i], dim=1)
        if y is not None:
            treeClass_true.extend(y[i].view(-1).tolist())
        treeClass_pred.extend(pred_classes.view(-1).tolist())

    TC = [treeClass_true, treeClass_pred]

    return TC


def evaluate(model, device, dataloader, targets, criterion=None, weight=None):

    model.eval()

    score_param_index = get_scoring_func_param_index(targets)
    if (weight is None) and (criterion is not None):
        weight = [1.0] * len(criterion)
    if criterion is not None:
        assert len(weight) == len(criterion)

    treeClass_pred, treeClass_true = None, None

    labeled = dataloader.dataset.labeled
    if score_param_index[0] is not None:
        treeClass_pred, treeClass_true = [], [] if labeled else None

    valid_loss = 0
    valid_n_iter = 0

    with torch.no_grad():
        for data in tqdm(dataloader):
            if labeled:
                x, y = data
                if not isinstance(y, (list, tuple)):
                    y = [y]
                x, y = x.to(device), [t.to(device) for t in y]
            else:
                x = data
                y = None
                x = x.to(device)

            x = x.reshape(x.shape[0], 1, 3072)
            outputs = model(x)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]

            if labeled and (criterion is not None):
                y[0] = y[0].long()
                loss = [
                    w * (c(o, gt) if o.size(1) ==
                         1 else c(o, gt.squeeze(1)))
                    for c, o, gt, w in zip(criterion, outputs, y, weight)
                ]
                loss = sum(loss)

            TC = update_prediction_data(
                score_param_index,
                y, outputs,
                [treeClass_true, treeClass_pred],
            )
            treeClass_true, treeClass_pred = TC

            if labeled and (criterion is not None):
                valid_loss += loss.item()
            valid_n_iter += 1

    # mean loss
    mean_loss = valid_loss / max(valid_n_iter, 1)

    # metrics
    if treeClass_pred is not None:
        treeClass_pred = np.array(treeClass_pred, dtype=np.int32)
    if treeClass_true is not None:
        treeClass_true = np.array(treeClass_true, dtype=np.int32)

    if labeled:
        metrics = scoreF.scorePerformance(
            treeClass_pred, treeClass_true
        )
    else:
        return (treeClass_pred)

    preds = (treeClass_pred)
    return mean_loss, metrics, preds
