from __future__ import annotations

import torch

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

class WeightedFalseClassPenaltyLogLoss(torch.nn.Module):
    """
    Generalized Cross-Entropy loss with additional false class penalties.
    Weights are a Tensor of shape (num_classes, num_classes), representing the weights of loss penalties for a given prediction.
    Diagonal entries give weights to standard cross-entropy loss terms log(p).
    Off-diagonal entries give weights to False-class log loss terms log(1-p), adding additional penalties for high false class prediction scores.
    Takes logits as inputs.
    """
    def __init__(self, weight_matrix_A, weight_matrix_B, normalization="none", mhh_weights=(1.0,1.0), loss_components_dict=None, device=None, **kwargs):
        super(WeightedFalseClassPenaltyLogLoss, self).__init__()

        print("signal vs. background matrix:")
        print(weight_matrix_A)
        print("kappa lambda vs. kappa lambda matrix:")
        print(weight_matrix_B)

        self.normalization = normalization

        print("normalized matrices:")
        self.weight_matrix_A = self.normalize_weight_matrix(weight_matrix_A)
        self.weight_matrix_B = self.normalize_weight_matrix(weight_matrix_B)

        self.loss_components_dict = loss_components_dict # dict containing various combinations of loss components of the weight matrix one wants to calculate in addition.
        self.device = device
        self.mhh_max, self.mhh_min = mhh_weights

    def forward(self, logits, targets, **kwargs):
        # calculate main loss
        self.mhh_weight(**kwargs)
        logits = self.align_logits_to_targets(logits, targets, **kwargs)
        loss = self.calculate_loss(logits, targets, **kwargs)

        # calculate extra losses that dont contribute to training
        if self.loss_components_dict is not None:
            loss_dict = self.get_weight_matrix_loss_components(logits, targets)
            return loss, loss_dict

        return (loss,)
    
    def mhh_weight(self, **kwargs):
        continuous_features = kwargs.get('cont_input')
        mhh = torch.sqrt(continuous_features[:,41]**2 - continuous_features[:,42]**2 - continuous_features[:,43]**2 - continuous_features[:,44]**2).unsqueeze(1)
        self.mhh_weights = self.get_piecewise_weights(mhh, self.mhh_max, self.mhh_min)

    
    def get_piecewise_weights(self, mhh_value, max_weight, min_weight):
        """
        Applies piecewise weighting:
        - x <= 350: max_weight
        - 350 < x < 500: linear transition
        - x >= 500: min_weight
        """
        # 1. Constant high weight for the start
        weights = torch.full_like(mhh_value, max_weight)
        
        # 2. Linear interpolation for the middle section (350 to 500)
        # Formula: high - (dist_from_350 / total_range) * (high - low)
        mid_mask = (mhh_value > 350) & (mhh_value < 500)
        linear_step = (mhh_value - 350) / (500 - 350)
        weights = torch.where(mid_mask, max_weight - (linear_step * (max_weight - min_weight)), weights)
        
        # 3. Constant low weight for everything 500 and above
        weights = torch.where(mhh_value >= 500, torch.tensor(min_weight).to(mhh_value.device), weights)
        return weights
    
    def normalize_weight_matrix(self, matrix):
        # sum of all elements in the entire matrix
        if self.normalization == 'global_sum':
            total_sum = torch.sum(torch.abs(matrix))            
            if total_sum == 0:
                matrix = matrix
                print(matrix)
            else:
                matrix = matrix / total_sum
                print(matrix)
        
        # divide all elements by the largest entry in the matrix
        elif self.normalization == 'max_norm':
            max_val = torch.max(torch.abs(matrix))
            if max_val == 0:
                matrix = matrix
                print(matrix)
            else:
                matrix = matrix / max_val
                print(matrix)

        # no normalization
        elif self.normalization == 'none':
            matrix = matrix
            print(matrix)
            
        else:
            raise ValueError("normalization must be 'global_sum', 'max_norm' or 'none'")
        
        return matrix
    
    def align_logits_to_targets(self, logits, targets, **kwargs):

        if logits.shape[1] == targets.shape[1]:
            return logits
        
        start_idx = kwargs.get('start_idx')
        end_idx = kwargs.get('end_idx')
        
        group_logit = torch.logsumexp(logits[:, start_idx:end_idx], dim=1, keepdim=True)
    
        # Stitch the logits back together
        return torch.cat([logits[:,:start_idx],group_logit], dim=1)   
    
    def calculate_loss(self, logits, targets, **kwargs):
        
        start_idx = kwargs.get('start_idx')
        if start_idx or self.weight_matrix_B is None:
            weight_matrix = self.weight_matrix_A
        else:
            weight_matrix = self.weight_matrix_B

        # calculate log losses
        true_class_log_prob = self.true_class_log_probabilities(logits)
        false_class_log_prob = self.false_class_log_probabilities(logits)

        # if torch.isnan(true_class_log_prob).any():
        #     print("NaN detected in true_class_log_prob!")   
        # if torch.isinf(true_class_log_prob).any():
        #     print("Inf detected in true_class_log_prob!")

        # if torch.isnan(false_class_log_prob).any():
        #     print("NaN detected in false_class_log_prob!")
        # if torch.isinf(false_class_log_prob).any():
        #     print("Inf detected in false_class_log_prob!")

        # pick out the row of the class
        weight_rows = targets @ weight_matrix 

        # calculate loss of target class
        true_class_loss = (targets * self.mhh_weights * weight_rows * true_class_log_prob).sum(dim=1)  # batch of only target loss

        # calculate loss of false classes
        false_classes_mask = (1 - targets)  # 1 where not true class
        false_class_loss = (false_classes_mask * self.mhh_weights * weight_rows * false_class_log_prob).sum(dim=1) # batch of false classes loss

        # add loss penalties
        loss_per_sample = -(true_class_loss + false_class_loss)
        loss = loss_per_sample.mean()
        return loss
    
    def get_weight_matrix_loss_components(self, logits, targets, **kwargs):
        """
        Method to compute components of the loss. Ex.: for weight matrix:
        [[1,2,3], [4,5,6], [7,8,9]]
        return the first row contribution, second row contribution, etc.
        """

        def weight_matrix_component_selector(mode, idx=None):
            """
            Selects a part of the weight matrix and returns a new matrix of the same shape
            with only the selected entries preserved; all others are zero.
            """
            selected_components = torch.zeros_like(self.weight_matrix)
            selected_components = selected_components.to(self.device)

            if mode == "row":
                selected_components[idx, :] = self.weight_matrix[idx, :]

            elif mode == "column":
                selected_components[:, idx] = self.weight_matrix[:, idx]

            elif mode == "element":
                i, j = idx
                selected_components[i, j] = self.weight_matrix[i, j]

            elif mode == "weighted_cross_entropy":
                diag_vals = self.weight_matrix.diagonal()
                selected_components.fill_diagonal_(diag_vals)

            elif mode == "submatrix":
                r1, r2, c1, c2 = idx
                selected_components[r1:r2, c1:c2] = self.weight_matrix[r1:r2, c1:c2]

            elif mode == "cross_entropy":
                n, m = self.weight_matrix.shape
                selected_components = torch.eye(n,m)
            else:
                raise ValueError("Invalid mode")

            return selected_components

        non_contributing_losses = {}
        start_idx = kwargs.get('start_idx')
        end_idx = kwargs.get('end_idx')

        if start_idx is not None or end_idx is not None:
            matrix_type = 'group'
        else:
            matrix_type = 'kl'

        for key, value in self.loss_components_dict[matrix_type].items():
            reduced_weight_matrix = weight_matrix_component_selector(value['mode'], value['content'])
            reduced_weight_matrix = reduced_weight_matrix.to(self.device)
            non_contributing_losses[key] = self.calculate_loss(logits, targets, reduced_weight_matrix)

        return non_contributing_losses

    def true_class_log_probabilities(self, logits):
        """
        Computes a numerically safe, equivalent form of log(p) using torch.nn.functional.log_softmax
        """
        true_class_log_probability = torch.log_softmax(logits, dim=1)

        return true_class_log_probability

    def false_class_log_probabilities(self, logits):
        
        log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
        one_minus_p = -torch.expm1(log_probabilities)
        false_class_log_probability = torch.log(one_minus_p.clamp(min=1e-10))

        return false_class_log_probability
    
    # old hardcoded loss for reference
    # probabilities = torch.nn.functional.softmax(pred, dim=1) # output probabilities
    # true_class_log_probabilities = torch.log(probabilities + 1e-12) # log probability of true class
    # false_class_log_probabilities = torch.log(1 - probabilities + 1e-12) # log probability of false classes

    # # pick out the row of the class
    # weight_rows = targets @ weight_matrix 

    # # calculate loss of target class
    # true_class_loss = (targets * weight_rows * true_class_log_probabilities).sum(dim=1)  # batch of only target loss

    # # calculate loss of false classes
    # false_classes_mask = 1 - targets  # 1 where not true class
    # false_class_loss = (false_classes_mask * weight_rows * false_class_log_probabilities).sum(dim=1) # batch of false classes loss

    # # add loss penalties
    # loss_per_sample = -(true_class_loss + false_class_loss) 
    # loss = loss_per_sample.mean() # reduce to scalar