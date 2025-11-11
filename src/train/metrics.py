import torch
import sklearn.metrics


def rates_binary(target, pred, threshold=0.5):
    tp = torch.sum((target == 1) & (pred > threshold))
    tn = torch.sum((target == 0) & (pred < threshold))
    fp = torch.sum((target == 0) & (pred > threshold))
    fn = torch.sum((target == 1) & (pred < threshold))
    return tp, tn, fp, fn

def rates_multiclass(targets, predictions, weights=None):
    """
    Calculate tp, tn, fp, fn for a 'Multi Label Classifier' by utilizing the *targets and *predictions* tensors.
    If a *weights* of events is given, this is applied to the rates calculation.

    Args:
        targets (_type_): _description_
        pred (_type_): _description_
        weights (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # get predicted class by ceil arg max index and floor everything else
    _prediction = torch.zeros_like(predictions)
    _predicted_class = torch.argmax(predictions, dim=1)
    # index enables to set values at specific indices by using a source
    _prediction = _prediction.scatter_(dim=1, index=_predicted_class.unsqueeze(1), value=1)

    # bring weights in correct form for broadcasting
    if weights is None:
        weights = torch.ones(targets.shape[0])
    weights = weights.reshape(-1, 1)

    # calculate tp, tn, fp, fn for all classes
    tp_diag = torch.sum( torch.logical_and( targets == 1, _prediction == 1) * weights, dim = 0)
    tn_rest = torch.sum(torch.logical_and( targets == 0, _prediction == 0) * weights, dim = 0)
    fp_per_column = torch.sum(torch.logical_and( targets == 0, _prediction == 1) * weights, dim = 0)
    fn_per_line = torch.sum(torch.logical_and( targets == 1, _prediction == 0) * weights, dim = 0)

    # stack rates into tensor, slice by truth class index
    # returns tp, tn, fp, fn for each class in order
    rates = torch.stack((tp_diag, tn_rest, fp_per_column, fn_per_line), dim=1)
    return rates

def rates_from_matrix(confusion_matrix, class_idx):
    tp = confusion_matrix[class_idx, class_idx]
    fn = torch.sum(confusion_matrix[class_idx, :]) - tp
    fp = torch.sum(confusion_matrix[:, class_idx]) - tp
    tn = torch.sum(confusion_matrix) - (tp + fn + fp)
    return tp, tn, fp, fn

# def precision(tp, tn, fp, fn):
#     return (tp + tn) / (tp + tn + fp + fn)

def precision(tp, fp):
    # precision measures the model's ability to identify instances of a particular class correctly
    return tp / (tp + fp)

def sensitivity(tp, fn):
    # recall measures the model's ability to identify all instances of a particular class.
    return tp / (tp + fn)

def recall(tp, fn):
    return sensitivity(tp, fn)

def calculate_metrics(target, pred, label=None, weights=None, normalize=False):
    metrics = {}
    rates = rates_multiclass(target, pred, weights=weights)
    num_cls = target.shape[-1]
    for c in range(num_cls):
        tp, tn, fp, fn = rates[c]

        # TODO wrong definition!
        if normalize:
            raise NotImplementedError("Normalization of metrics not implemented yet.")
            num_true = tp + fn
            tp, tn, fp, fn = tp / num_true, tn / num_true, fp / num_true, fn / num_true

        _precision = precision(tp, fp,)
        _sensitivity = sensitivity(tp, fn)
        _f1_score = F1_score(_precision, _sensitivity)
        if label is not None:
            c = label[c]
        metrics[c] = {
            "tp": tp.item(),
            "tn": tn.item(),
            "fp": fp.item(),
            "fn": fn.item(),
            "precision": _precision.item(),
            "sensitivity": _sensitivity.item(),
            "f1_score": _f1_score.item(),
            }
    return metrics

# TODO: use output of calculate metrics to get macro metrics, do not recalculate
def calculate_macro_metrics(metrics):
    pass

def F1_score(precision, recall):
    return 2 * (precision * recall) / (precision + recall)


def auc(fp, tp):
    return sklearn.metrics.auc(fp, tp)

def matthews_correlation_coefficient_binary(tp, tn, fp, fn):
    numerator = (tp * tn) - (fp * fn)
    denominator = torch.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return torch.tensor(0.0)
    return numerator / denominator
