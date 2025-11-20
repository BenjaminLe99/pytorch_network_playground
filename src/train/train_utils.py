import torch
import plotting
import metrics


def training(model, loss_fn, optimizer, sampler, device):
    optimizer.zero_grad()

    cont, cat, targets = sampler.sample_batch(device=device)
    targets = targets.to(torch.float32)

    #pred = model((cat,cont))
    pred = model(categorical_inputs=cat, continuous_inputs=cont)

    loss = loss_fn(pred, targets.reshape(-1,3))
    loss.backward()
    optimizer.step()
    return loss, (pred, targets)

def validation(model, loss_fn, sampler, device):
    with torch.no_grad():
        # run validation every x steps
        val_loss = []
        model.eval()
        predictions = []
        truth = []
        weights = []

        for uid, validation_batch_generator in sampler.get_dataset_batch_generators(batch_size=sampler.batch_size, device=device).items():
            dataset_losses = []
            for cont, cat, tar in validation_batch_generator:
                cat, cont, tar = cat, cont, tar
                val_pred = model(categorical_inputs=cat, continuous_inputs=cont)

                loss = loss_fn(val_pred, tar.reshape(-1, 3))

                dataset_losses.append(loss)

                #predictions.append(val_pred).cpu()
                predictions.append(torch.softmax(val_pred, dim=1).cpu())
                truth.append(tar.cpu())
                weights.append(torch.full(size=(val_pred.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())
            # create event based weight tensor for dataset

            average_val = sum(dataset_losses) / len(dataset_losses) * sampler[uid].relative_weight
            val_loss.append(average_val)

        final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)
        model.train()

        truth = torch.concatenate(truth, dim=0)
        predictions = torch.concatenate(predictions, dim=0)
        weights = torch.flatten(torch.concatenate(weights, dim=0))
        return final_validation_loss, (predictions, truth, weights)


def log_metrics(tensorboard_inst, iteration_step, sampler_output, target_map, mode="train", **data):
    # general logging
    if (loss := data.get("loss")) is not None:
        tensorboard_inst.log_loss({mode: loss}, step=iteration_step)

    if (lr := data.get("lr")) is not None:
        tensorboard_inst.log_lr(lr, step=iteration_step)

    pred, tar, weights = sampler_output

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
