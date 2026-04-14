import torch
import plotting
import metrics
import train_config

functions = {}

def register(fn):
    # helper to register functions in pool of functions
    functions[fn.__name__] = fn
    return fn

@register
def training_default(model, loss_fn, optimizer, target_map, strength_param, sampler, device):
    optimizer.zero_grad()

    cont, cat, targets = sampler.sample_batch(device=device)
    logits = model(categorical_inputs=cat, continuous_inputs=cont)

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
    group_loss, *other_group_losses = loss_fn(logits, group_targets, start_idx=start_idx, end_idx=end_idx)
    kl_loss, *other_kl_losses = loss_fn(kl_logits, kl_targets)

    # safely extract dicts incase None was passed in the loss components dict.
    dict_group = other_group_losses[0] if other_group_losses else {}
    dict_kl = other_kl_losses[0] if other_kl_losses else {}

    other_losses = [dict_group | dict_kl]

    loss = (strength_param * group_loss) + kl_loss
    loss.backward()

    for name, p in model.named_parameters():
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print("BAD GRAD:", name)
            from IPython import embed;embed(header=" string - 26 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train_utils.py")
    
    optimizer.step()

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
def validation_default(model, loss_fn, target_map, strength_param, sampler, device):
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
                group_loss, *other_group_losses = loss_fn(logits, group_targets, start_idx=start_idx, end_idx=end_idx)
                kl_loss, *other_kl_losses = loss_fn(kl_logits, kl_targets)

                dict_group = other_group_losses[0] if other_group_losses else {}
                dict_kl = other_kl_losses[0] if other_kl_losses else {}

                other_losses = [dict_group | dict_kl]
                loss = (strength_param * group_loss) + kl_loss
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


def log_metrics(tensorboard_inst, iteration_step, sampler_output, target_map, mode="train", **data):
    # general logging
    if (loss := data.get("loss")) is not None:
        tensorboard_inst.log_loss({mode: loss}, step=iteration_step)

    if (other_loss := data.get("other_loss")):
        for key, value in other_loss[0].items():
            tensorboard_inst.log_loss({key: value.item()}, step=iteration_step)

    if (lr := data.get("lr")) is not None:
        tensorboard_inst.log_lr(lr, step=iteration_step)

    pred, tar, weights = sampler_output

    if float(torch.sum(torch.isnan(pred)).detach()) > 0:
        from IPython import embed;embed(header=" string - 98 in /afs/desy.de/user/l/lebenjam/Master/neuralnetwork/src/train/train_utils.py")

    # network prediction plot
    pred_fig, pred_ax = plotting.network_predictions(
        tar,
        pred,
        target_map
    )
    tensorboard_inst.log_figure(f"{mode} node output", pred_fig, step=iteration_step)

    # confusion matrix plot
    c_mat_fig, c_mat_ax, c_mat = plotting.confusion_matrix(
        tar,
        pred,
        target_map,
        sample_weight=weights,
        normalized="true"
    )
    tensorboard_inst.log_figure(f"{mode} confusion matrix", c_mat_fig, step=iteration_step)

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

    tensorboard_inst.log_precision(_metrics, step=iteration_step, mode=mode)
    tensorboard_inst.log_sensitivity(_metrics, step=iteration_step, mode=mode)

training_fn = functions.get(f"training_{train_config.config['training_fn']}")
validation_fn = functions.get(f"validation_{train_config.config['validation_fn']}")
