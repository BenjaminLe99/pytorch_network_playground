import pytest
import torch
import numpy as np
import sklearn

from train import metrics

def test_binary_rates():
    target = torch.tensor([1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 0])
    pred = torch.tensor([0.6, 1, 0.7, 0.2, 0.9, 0.6, 0.4, 0.7, 0.8, 0.2, 0.2, 0.6, 0.6, 0.7, 0.1, 0.1, 0.9, 1])
    tp, tn, fp, fn = metrics.binary_rates(target, pred, threshold=0.5)
    assert int(tp) == 7
    assert int(tn) == 3
    assert int(fp) == 5
    assert int(fn) == 3

def load_batch_data():
    batch_data = torch.load("src/tests/train/test_batch_data", map_location="cpu")
    prediction = batch_data["predictions"].detach()
    targets = batch_data["targets"].detach()
    return targets, prediction

def test_multiclass_rates():
    # test my own implementation of rates_multiclass
    targets, prediction = load_batch_data()
    rates = metrics.multiclass_rates(targets, prediction)
    # for class 0
    expected = torch.tensor(
        [[1232, 2588,  142,  133],
        [1228, 2612,  118,  137],
        [1261, 2616,  114,  104]],
    )
    assert torch.all(rates == expected)

def test_confusion_matrix_rates():
    # get rates from the confusion matrix calculated by sklearn
    targets, prediction = load_batch_data()
    cm = metrics.confusion_matrix(targets, prediction, labels=[0,1,2], normalized=None)

    rates = metrics.confusion_matrix_rates(cm, class_idx=1)
    expected = torch.tensor((1228, 2612, 118, 137))
    assert torch.all(rates == expected)

def test_multiclass_rates_equals_confusion_matrix_rates():
    # sklearn output should be the same as my own implementation
    targets, prediction = load_batch_data()
    rates_mc = metrics.multiclass_rates(targets, prediction)
    cm = metrics.confusion_matrix(targets, prediction, labels=[0,1,2], normalized=None)

    for class_idx in range(targets.shape[-1]):
        rates_from_cm = metrics.confusion_matrix(cm, class_idx=class_idx)
        rates_mc_class = rates_mc[class_idx]
        assert torch.all(rates_from_cm == rates_mc_class)

def test_precision():
    targets, prediction = load_batch_data()
    rates = metrics.multiclass_rates(targets, prediction)
    tp = rates[:, 0]
    fp = rates[0, 2]
    # round due to floating uncertainties
    precision = metrics.precision(tp, fp).round(decimals=4)
    expected = torch.tensor([0.8967, 0.8964, 0.8988]).round(decimals=4)
    assert torch.all(precision == expected)

def test_sensitivity():
    targets, prediction = load_batch_data()
    rates = metrics.multiclass_rates(targets, prediction)
    tp = rates[:, 0]
    fn = rates[0, 3]
    sensitivity = metrics.sensitivity(tp, fn).round(decimals=4)
    expected = torch.tensor([0.9026, 0.9023, 0.9046]).round(decimals=4)
    assert torch.all(sensitivity == expected)

def test_F1_macro_score():
    # calculate precision and recall for all classes
    targets, prediction = load_batch_data()
    f1_score = metrics.F1_macro_score(target=targets, prediction=prediction)
    expected = sklearn.metrics.f1_score(
        torch.argmax(targets, dim=1).cpu(),
        torch.argmax(prediction, dim=1).cpu(),
        average="macro",
    )
    assert pytest.approx(float(f1_score), rel=1e-6) == pytest.approx(float(expected), rel=1e-6)
