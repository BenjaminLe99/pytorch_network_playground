from __future__ import annotations

import torch





class SignalEfficiency(torch.nn.Module):
    def __init__(self, target_map, total_weights, device, *args, **kwargs):
        super().__init__()
        self.target_map = target_map
        self.s_cls = self.target_map["hh"]
        self.total_weights = total_weights
        self.device = device

    def s(self, prediction, truth, weight):
        # true positive weighted by signal node
        return weight * torch.sum(truth[:, self.s_cls] * prediction[:, self.s_cls])

    def b(self, prediction, truth, weight):
        # false negative
        return weight * torch.sum((1 - truth[:, self.s_cls]) * prediction[:, self.s_cls])

    def _loss_v1(self, prediction, truth, weight):
        s_weight = weight["signal"]
        b_weight = weight["background"]
        s = self.s(prediction, truth, s_weight)
        b = self.b(prediction, truth, b_weight)
        loss = s / (s + b)**0.5
        return 1 / loss

    def _loss_asm(self, prediction, truth, weight):
        s_weight = weight["signal"]
        b_weight = weight["background"]
        s = self.s(prediction, truth, s_weight)
        b = self.b(prediction, truth, b_weight)
        sig_b = 0 # TODO wie berechnen?

        ln = torch.log

        N = s + b
        B = b**2
        Si = sig_b**2

        first_term = N * ln(N * (b + Si) / B + N*Si)
        second_term = (B / Si) * ln(1 + (Si * s)/(B + b*Si))

        full_term = 2 * (first_term - second_term)
        return torch.sqrt(full_term)


    def forward(self, prediction, truth, weight, threshold = 0.5):
        from IPython import embed; embed(header = " line: 59 in loss.py")


        return None

class ZScore(torch.nn.Module):
    """
    Calculates a surrogate loss by calculating signal efficiency for given *signal_cls_idx*.
    This discrete value is used as an coefficient to amplify a given *base_loss_inst*, extending
    this discrete values to be differentiable.

    Idea come from https://arxiv.org/pdf/2412.09500

    Args:
        signal_cls_idx (int): Index of the signal class in the prediction/target tensors.
        base_loss_inst (torch.nn.Module): Base loss function to be amplified.
    """

    def __init__(self, signal_cls_idx, base_loss_inst):
        super().__init__()
        self.signal_cls_idx = signal_cls_idx
        self.eps = 1
        self.base_loss_inst = base_loss_inst

    def split_signal_and_background(self, pred, truth):
        signal_events_idx = torch.min(torch.argwhere(truth[:,self.signal_cls_idx]))
        signal_truth = truth[signal_events_idx:]
        background_truth = truth[:signal_events_idx]
        signal_pred = pred[signal_events_idx:]
        background_pred = pred[:signal_events_idx]
        return signal_pred, signal_truth, background_pred, background_truth

    def get_slices(self, idx):
        start = torch.tensor([0] + idx[:-1]).cumsum(0).tolist()
        end = torch.tensor(idx).cumsum(0).tolist()
        return start, end

    def signal_score(self, signal_prediction, signal_truth, weight, idx):
        start_idx, end_idx = self.get_slices(idx["signal"])

        signal_scores = []
        for current_signal_idx, (l, r) in enumerate(zip(start_idx, end_idx)):
            # select sub batches
            sub_targets = signal_truth[l:r]
            sub_predictions = signal_prediction[l:r]

            pred_cls = torch.argmax(sub_predictions, dim=1)
            truth_cls = torch.argmax(sub_targets, dim=1)

            # false negative is any bg classified as signal
            # TODO: differitiazion between multiple signal will not work with this code
            fn_signal = torch.sum(pred_cls != truth_cls)

            # calculate weighted fraction of correctly classified signal events
            num_signal = len(sub_targets)
            fraction = (num_signal - fn_signal)/num_signal
            score = fraction * weight["signal"][current_signal_idx]
            signal_scores.append(score)
        return torch.tensor(signal_scores)

    def background_scores(self, background_pred, background_truth, weight, idx):
        """ calculates for each background sample the sqrt of the score based on misclassification into signal"""
        start_idx, end_idx = self.get_slices(idx["background"])

        bg_scores = []
        # HINT: this score does not differentiate between different background types
        for current_bg_idx, (l, r) in enumerate(zip(start_idx, end_idx)):
            sub_targets = background_truth[l:r]
            sub_predictions = background_pred[l:r]

            pred_cls = torch.argmax(sub_predictions, dim=1)

            # false negative in the case of background is signal classified as background, thus being signal_cls
            # TODO maybe include misclassification into other background classes as well?
            fn_signal_as_bg = torch.sum(pred_cls == self.signal_cls_idx)

            num_bg = len(sub_targets)
            rate_of_miscl_into_signal  = (fn_signal_as_bg) / num_bg
            score = torch.sqrt(
                rate_of_miscl_into_signal * weight["background"][current_bg_idx] + self.eps
                )
            bg_scores.append(score)
        return torch.tensor(bg_scores)

    def forward(self, pred, truth, weight, idx):
        signal_predictions, signal_truth, background_prediction, background_truth = self.split_signal_and_background(pred, truth)
        # separate signal and background

        signal_fraction_score = torch.sum(self.signal_score(signal_predictions, signal_truth, weight, idx))
        background_fraction_score = torch.sum(self.background_scores(background_prediction, background_truth, weight, idx))

        max_score = torch.sum(torch.tensor(weight["signal"])) / self.eps**0.5
        # print(signal_fraction_score, background_fraction_score, max_score)
        coefficient = (max_score - signal_fraction_score) / background_fraction_score
        loss = self.base_loss_inst(pred, truth)
        if torch.isnan(coefficient) or torch.isnan(loss):
            from IPython import embed; embed(header="Either coefficient or loss is NaN")
        return loss * coefficient


class WeightedCrossEntropy(torch.nn.CrossEntropyLoss):
    def forward(self, input, target, weight: torch.Tensor | None = None):
        # save original reduction mode
        reduction = self.reduction
        if weight is not None:
            self.reduction = "none"
            loss = super().forward(input, target)
            self.reduction = reduction

            # dot product is only defined for flat tensors, so flatten
            loss = torch.flatten(loss)
            weight = torch.flatten(weight)
            loss = torch.dot(loss, weight)
            if self.reduction == "mean":
                loss = loss / torch.sum(weight)
        else:
            loss = super().forward(input, target)
        return loss


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2, alpha=None, reduction='mean', task_type='binary', num_classes=None):
        """
        Taken from https://github.com/itakurah/Focal-loss-PyTorch/blob/main/focal_loss.py

        Unified Focal Loss class for binary, multi-class, and multi-label classification tasks.
        :param gamma: Focusing parameter, controls the strength of the modulating factor (1 - p_t)^gamma
        :param alpha: Balancing factor, can be a scalar or a tensor for class-wise weights. If None, no class balancing is used.
        :param reduction: Specifies the reduction method: 'none' | 'mean' | 'sum'
        :param task_type: Specifies the type of task: 'binary', 'multi-class', or 'multi-label'
        :param num_classes: Number of classes (only required for multi-class classification)
        """
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.task_type = task_type
        self.num_classes = num_classes

        # Handle alpha for class balancing in multi-class tasks
        if task_type == 'multi-class' and alpha is not None and isinstance(alpha, (list, torch.Tensor)):
            assert num_classes is not None, "num_classes must be specified for multi-class classification"
            if isinstance(alpha, list):
                self.alpha = torch.Tensor(alpha)
            else:
                self.alpha = alpha

    def forward(self, inputs, targets):
        """
        Forward pass to compute the Focal Loss based on the specified task type.
        :param inputs: Predictions (logits) from the model.
                    Shape:
                        - binary/multi-label: (batch_size, num_classes)
                        - multi-class: (batch_size, num_classes)
        :param targets: Ground truth labels.
                        Shape:
                        - binary: (batch_size,)
                        - multi-label: (batch_size, num_classes)
                        - multi-class: (batch_size,)
        """
        if self.task_type == 'binary':
            return self.binary_focal_loss(inputs, targets)
        elif self.task_type == 'multi-class':
            return self.multi_class_focal_loss(inputs, targets)
        elif self.task_type == 'multi-label':
            return self.multi_label_focal_loss(inputs, targets)
        else:
            raise ValueError(
                f"Unsupported task_type '{self.task_type}'. Use 'binary', 'multi-class', or 'multi-label'.")

    def binary_focal_loss(self, inputs, targets):
        """ Focal loss for binary classification. """
        probs = torch.sigmoid(inputs)
        targets = targets.float()

        # Compute binary cross entropy
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weighting
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_class_focal_loss(self, inputs, targets):
        """ Focal loss for multi-class classification. """
        if self.alpha is not None:
            alpha = self.alpha.to(inputs.device)

        # Convert logits to probabilities with softmax
        probs = torch.nn.functional.softmax(inputs, dim=1)

        # One-hot encode the targets
        targets_one_hot = torch.nn.functional.one_hot(targets, num_classes=self.num_classes).float()

        # Compute cross-entropy for each class
        ce_loss = -targets_one_hot * torch.log(probs)

        # Compute focal weight
        p_t = torch.sum(probs * targets_one_hot, dim=1)  # p_t for each sample
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided (per-class weighting)
        if self.alpha is not None:
            alpha_t = alpha.gather(0, targets)
            ce_loss = alpha_t.unsqueeze(1) * ce_loss

        # Apply focal loss weight
        loss = focal_weight.unsqueeze(1) * ce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss

    def multi_label_focal_loss(self, inputs, targets):
        """ Focal loss for multi-label classification. """
        probs = torch.sigmoid(inputs)

        # Compute binary cross entropy
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        # Compute focal weight
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_weight = (1 - p_t) ** self.gamma

        # Apply alpha if provided
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            bce_loss = alpha_t * bce_loss

        # Apply focal loss weight
        loss = focal_weight * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
