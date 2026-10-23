import torch
import plotting
import metrics
import train_config
import math
from contextlib import contextmanager
import time
import bisect
from torch.optim.swa_utils import AveragedModel

TIMING_ENABLED = False

@contextmanager
def time_block(name):
    if not TIMING_ENABLED:
        yield  # Passes right through without timing
        return

    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"\n--- TIMING REPORT ---")
    print(f"Block : {name}")
    print(f"Time  : {elapsed:.4f} seconds")
    print(f"-----------------------\n")

functions = {}

def register(fn):
    # helper to register functions in pool of functions
    functions[fn.__name__] = fn
    return fn

@register
def training_default(model, ema_model, ema_adaptive_fn, loss_fn, optimizer, target_map, strength_param, only_one_weightmatrix, sampler, device):
    optimizer.zero_grad()

    cont, cat, targets = sampler.sample_batch(device=device)
    logits = model(categorical_inputs=cat, continuous_inputs=cont)

    if 'hh' not in target_map and only_one_weightmatrix == False:
        # get indices for the kappa lambda classes
        group_indices = [value for key, value in target_map.items() if key in ['kl0','kl1','kl2','kl5']]
        start_idx = min(group_indices)
        end_idx = max(group_indices) + 1

        # prep for background vs signal loss
        bg_targets = targets[:,:start_idx]
        sig_targets = targets[:,start_idx:end_idx].sum(dim=1, keepdim=True)
        group_targets = torch.cat([bg_targets,sig_targets], dim=1)

        # prep for kappa lambda vs kappa lambda loss
        kl_targets = targets[:,start_idx:end_idx]
        kl_logits = logits[:,start_idx:end_idx]

        # calculate the loss for background vs signal and kappa lambda vs kappa lambda
        group_loss, *other_group_losses = loss_fn(logits, group_targets, cont_input=cont, start_idx=start_idx, end_idx=end_idx)
        kl_loss, *other_kl_losses = loss_fn(kl_logits, kl_targets, cont_input=cont)

        # safely extract dicts incase None was passed in the loss components dict.
        dict_group = other_group_losses[0] if other_group_losses else {}
        dict_kl = other_kl_losses[0] if other_kl_losses else {}

        other_losses = [dict_group | dict_kl]

        loss = group_loss + (strength_param * kl_loss)
    elif 'hh' in target_map or only_one_weightmatrix == True:
        loss, *other_losses = loss_fn(logits, targets, cont_input=cont)

    loss.backward()

    # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))  # fused norm, no real clipping
    # if not torch.isfinite(grad_norm):  # one sync per step
    #     for name, p in model.named_parameters():
    #         if p.grad is not None and not torch.isfinite(p.grad).all():
    #             print("BAD GRAD:", name)
    #     raise RuntimeError("non-finite gradient")  # or embed() when running interactively

    # for name, p in model.named_parameters():
    #     if p.grad is not None and not torch.isfinite(p.grad).all():
    #         print("BAD GRAD:", name)
    #         from IPython import embed;embed(header=" string - 26 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train_utils.py")

    optimizer.step()

    if ema_model:
        if ema_adaptive_fn:
            ema_adaptive_fn.step_loss(loss)
        ema_model.update_parameters(model)

    if other_losses:
        return loss, (logits, targets), other_losses

    return loss, (logits, targets)

@register
def training_sam(model, loss_fn, optimizer, sampler, device):
    optimizer.zero_grad()

    cont, cat, targets = sampler.sample_batch(device=device)
    pred = model(categorical_inputs=cat, continuous_inputs=cont)

    loss = loss_fn(pred, targets)
    loss.backward()
    optimizer.first_step(zero_grad=True)

    # second forward step with disabled bachnorm running stats in second forward step
    optimizer.disable_running_stats(model)
    pred_2 = model(categorical_inputs=cat, continuous_inputs=cont)
    loss_fn(pred_2, targets).backward()


    optimizer.second_step(zero_grad=True)

    optimizer.enable_running_stats(model)  # <- this is the important line
    return loss, (pred, targets)

@register
def validation_default(model, loss_fn, target_map, strength_param, only_one_weightmatrix, sampler, device):
    with torch.no_grad():
        # run validation every x steps
        val_loss = []
        model.eval()
        predictions = []
        truth = []
        weights = []
        other_val_losses = {}

        for uid, validation_batch_generator in sampler.get_dataset_batch_generators(batch_size=sampler.batch_size, device=device).items():
            dataset_losses = []
            other_dataset_losses = {}
            for cont, cat, tar in validation_batch_generator:
                logits = model(categorical_inputs=cat, continuous_inputs=cont)

                if 'hh' not in target_map and only_one_weightmatrix == False:
                    # get indices for the kappa lambda classes
                    group_indices = [value for key, value in target_map.items() if key in ['kl0','kl1','kl2','kl5']]
                    start_idx = min(group_indices)
                    end_idx = max(group_indices) + 1

                    # prep for background vs signal loss
                    bg_targets = tar[:,:start_idx]
                    sig_targets = tar[:,start_idx:end_idx].sum(dim=1, keepdim=True)
                    group_targets = torch.cat([bg_targets,sig_targets], dim=1)

                    # prep for kappa lambda vs kappa lambda loss
                    kl_targets = tar[:,start_idx:end_idx]
                    kl_logits = logits[:,start_idx:end_idx]

                    # calculate the loss for background vs signal and kappa lambda vs kappa lambda
                    group_loss, *other_group_losses = loss_fn(logits, group_targets, cont_input=cont, start_idx=start_idx, end_idx=end_idx)
                    kl_loss, *other_kl_losses = loss_fn(kl_logits, kl_targets, cont_input=cont)

                    dict_group = other_group_losses[0] if other_group_losses else {}
                    dict_kl = other_kl_losses[0] if other_kl_losses else {}

                    other_losses = [dict_group | dict_kl]
                    loss = (strength_param * group_loss) + kl_loss
                elif 'hh' in target_map or only_one_weightmatrix == True:
                    loss, *other_losses = loss_fn(logits, tar, cont_input=cont)
                dataset_losses.append(loss)

                if other_losses:
                    for key, value in other_losses[0].items():
                        other_dataset_losses.setdefault(key, []).append(value)

                #predictions.append(logits).cpu()
                predictions.append(torch.softmax(logits, dim=1).cpu())
                truth.append(tar.cpu())
                weights.append(torch.full(size=(logits.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())
            # create event based weight tensor for dataset

            average_val = sum(dataset_losses) / len(dataset_losses) * sampler[uid].relative_weight
            val_loss.append(average_val)

            if other_dataset_losses:
                other_average_val = {}
                for key, value in other_dataset_losses.items():
                    other_average_val[key] = sum(value) / len(value) * sampler[uid].relative_weight
                    other_val_losses.setdefault(key, []).append(other_average_val[key])

        final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)

        if other_val_losses:
            for key, value in other_val_losses.items():
                other_val_losses[key] = sum(value).cpu()

            model.train()

            truth = torch.concatenate(truth, dim=0)
            predictions = torch.concatenate(predictions, dim=0)
            weights = torch.flatten(torch.concatenate(weights, dim=0))
            return final_validation_loss, (predictions, truth, weights), other_val_losses

        model.train()

        truth = torch.concatenate(truth, dim=0)
        predictions = torch.concatenate(predictions, dim=0)
        weights = torch.flatten(torch.concatenate(weights, dim=0))
        return final_validation_loss, (predictions, truth, weights)


def log_metrics(tensorboard_inst, iteration_step, sampler_output, target_map, metric_logging, mode="train", **data):
    # general logging
    if "loss" in metric_logging and (loss := data.get("loss")) is not None:
        with time_block("loss"):
            tensorboard_inst.log_loss({mode: loss}, step=iteration_step)

    if "other_loss" in metric_logging and (other_loss := data.get("other_loss")):
        with time_block("other loss"):
            for key, value in other_loss[0].items():
                tensorboard_inst.log_loss({key: value.item()}, step=iteration_step)

    if "lr" in metric_logging and (lr := data.get("lr")) is not None:
        with time_block("lr"):
            tensorboard_inst.log_lr(lr, step=iteration_step)

    pred, tar, weights = sampler_output

    if float(torch.sum(torch.isnan(pred)).detach()) > 0:
        from IPython import embed;embed(header=" string - 98 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train_utils.py")

    # network prediction plotting"):
    if "prediction" in metric_logging:
        with time_block("prediction"):
            pred_fig, pred_ax = plotting.network_predictions(
                tar,
                pred,
                target_map
            )
            tensorboard_inst.log_figure(f"{mode} node output", pred_fig, step=iteration_step)

    # confusion matrix plotx plotting"):
    if "confusion" in metric_logging:
        with time_block("confusion"):
            c_mat_fig, c_mat_ax, c_mat = plotting.confusion_matrix(
                tar,
                pred,
                target_map,
                sample_weight=weights,
                normalized="true"
            )
            tensorboard_inst.log_figure(f"{mode} confusion matrix", c_mat_fig, step=iteration_step)

    if "roc" in metric_logging:
        with time_block("ROC"):
            roc_fig, roc_ax = plotting.roc_curve(
                tar,
                pred,
                sample_weight=weights,
                labels=list(target_map.keys())
            )
            tensorboard_inst.log_figure(f"{mode} roc curve one vs rest", roc_fig, step=iteration_step)

    # TODO: metrics calculation
    _metrics = metrics.calculate_metrics(
        tar,
        pred,
        label=list(target_map.keys()),
        weights=weights,
    )
    if "precision" in metric_logging:
        with time_block("precision"):
            tensorboard_inst.log_precision(_metrics, step=iteration_step, mode=mode)
    if "sensitivity" in metric_logging:
        with time_block("sensitivity"):
            tensorboard_inst.log_sensitivity(_metrics, step=iteration_step, mode=mode)

training_fn = functions.get(f"training_{train_config.config['training_fn']}")
validation_fn = functions.get(f"validation_{train_config.config['validation_fn']}")

def validation_frequency(iteration, max_iteration):
    """Returns how often to validate based on current progress."""
    progress = iteration / max_iteration

    if progress < 0.5:
        return 500
    elif progress < 0.8:
        return 500
    else:
        return 500

def lr_multiplier(iteration, warmup_max, max_iterations, max_lr, min_lr):
    """Calculates the multiplier for the learning rate."""
    # The multiplier range we are scaling within
    min_multiplier = min_lr / max_lr
    multiplier_range = (max_lr - min_lr) / max_lr

    if iteration < warmup_max:
        # Linear Warmup: Scale from min_multiplier up to 1.0
        progress = float(iteration) / float(max(1, warmup_max))
        return min_multiplier + (progress * multiplier_range)

    # Cosine Annealing: Scale from 1.0 down to min_multiplier
    progress = float(iteration - warmup_max) / float(max(1, max_iterations - warmup_max))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

    return min_multiplier + (cosine_decay * multiplier_range)

def update_ema_batchnorm(ema_model, sampler, device, num_batches=30):
    ema_model.train() # Crucial: tells BatchNorm to update stats

    with torch.no_grad():
        for _ in range(num_batches):
            batch_cont, batch_cat, _ = sampler.sample_batch(device=device, shuffle_batch=True)
            # Dummy forward pass to update the statistics
            _ = ema_model(batch_cat, batch_cont)

class EMAMultiAvgFn:                      # constant decay
    def __init__(self, decay=0.998):
        self.decay = decay

    def get_window_size(self, step):
        # Convert alpha back to effective window size N = 1 / (1 - alpha)
        return 1.0 / (1.0 - self.decay)

    def __call__(self, step):
        return self.decay

class InverseStepEMAFn:
    def __init__(self, config):
        self.target_window = config["ema_inverse_target"]
        self.alpha_max = 1.0 - (1.0 / self.target_window)

        c = config["ema_inverse_c"]
        if isinstance(c, (int, float)):
            knots = [(0, float(c))]  # constant c, same as before
        else:
            knots = [(int(s), float(v)) for s, v in c]
        steps = [s for s, _ in knots]
        assert steps == sorted(steps), "c knots must be sorted by step"
        assert all(v > 1.0 for _, v in knots), "every c must be > 1"
        self.c_steps = steps
        self.c_vals = [v for _, v in knots]

    def get_c(self, step):
        # Linear interpolation between knots; constant outside the knot range
        if step <= self.c_steps[0]:
            return self.c_vals[0]
        if step >= self.c_steps[-1]:
            return self.c_vals[-1]
        i = bisect.bisect_right(self.c_steps, step) - 1
        s0, s1 = self.c_steps[i], self.c_steps[i + 1]
        v0, v1 = self.c_vals[i], self.c_vals[i + 1]
        return v0 + (v1 - v0) * (step - s0) / (s1 - s0)

    def get_window_size(self, step):
        if isinstance(step, torch.Tensor):
            step = step

        # Inverse step formula with step-dependent c: alpha = (1 + step) / (c(step) + step)
        alpha = (1.0 + step) / (self.get_c(step) + step)
        alpha = min(self.alpha_max, alpha)

        # Convert alpha back to effective window size N = 1 / (1 - alpha)
        return 1.0 / (1.0 - alpha)

    def __call__(self, step):
        return 1.0 - 1.0 / self.get_window_size(step)

# Usage:
# ema_model = AveragedModel(model, avg_fn=InverseStepEMAFn(target_window=2000.0))

class PiecewiseEMAFn:
    def __init__(self, config, total_steps):
        self.phase1_steps = config["ema_schedule"][0]
        self.phase1_size = config["ema_window_schedule"][0]
        self.phase2_steps = config["ema_schedule"][1]
        self.phase2_size = config["ema_window_schedule"][1]
        self.total_steps = total_steps
        self.phase3_size = config["ema_window_schedule"][2]

    def get_window_size(self, step):
        if isinstance(step, torch.Tensor):
            step = step

        # Phase 1: Static window
        if step < self.phase1_steps:
            return self.phase1_size
        # Phase 2: Linear ramp
        elif step < self.phase2_steps:
            # progress = (step - self.phase1_steps) / max(1, self.phase2_steps - self.phase1_steps)
            # return self.phase1_size + progress * (self.phase2_size - self.phase1_size)
            return self.phase2_size
        # Phase 3: Slow ramp
        else:
            # progress = (step - self.phase2_steps) / max(1, self.total_steps - self.phase2_steps)
            # return self.phase2_size + progress * (self.phase3_size - self.phase2_size)
            return self.phase3_size

    def __call__(self, step):
        return 1.0 - 1.0 / self.get_window_size(step)

class FastAveragedModel(AveragedModel):
    def __init__(self, model, alpha_fn):
        super().__init__(model)
        self.alpha_fn = alpha_fn
        self._n = 0
        self._ema_p = None

    @torch.no_grad()
    def update_parameters(self, model):
        if self._ema_p is None:
            self._ema_p = list(self.module.parameters())
            self._cur_p = list(model.parameters())
        if self._n == 0:
            torch._foreach_copy_(self._ema_p, self._cur_p)
        else:
            torch._foreach_lerp_(self._ema_p, self._cur_p, 1.0 - self.alpha_fn(self._n))
        self._n += 1
        self.n_averaged.fill_(self._n)

    def load_state_dict(self, *args, **kwargs):  # your plateau restore path
        out = super().load_state_dict(*args, **kwargs)
        self._n = int(self.n_averaged)
        return out