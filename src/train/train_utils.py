import torch
import plotting
import metrics
import train_config
from loss import ZScore
functions = {}

def register(fn):
    # helper to register functions in pool of functions
    functions[fn.__name__] = fn
    return fn

@register
def training_defaultV2(model, loss_fn, optimizer, sampler, device):
    optimizer.zero_grad()

    cont, cat, targets = sampler.sample_batch(device=device)

    #pred = model((cat,cont))
    pred = model(categorical_inputs=cat, continuous_inputs=cont)

    loss = loss_fn(pred, targets.reshape(-1,3))
    loss.backward()
    optimizer.step()
    return loss, (pred, targets)

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
def training_default(model, loss_fn, optimizer, sampler, device):
    optimizer.zero_grad()

    cont, cat, targets, weight, idx = sampler.sample_batch(device=device)
    pred = model(categorical_inputs=cat, continuous_inputs=cont)

    # loss = loss_fn(pred, targets.reshape(-1,3), weight, idx)
    loss = loss_fn(pred, targets.reshape(-1,3))
    loss.backward()
    optimizer.step()
    return loss, (pred, targets)

# @register
# def validation_default(model, loss_fn, sampler, device):
#     with torch.no_grad():
#         # run validation every x steps
#         val_loss = []
#         model.eval()
#         predictions = []
#         truth = []
#         weights = []

#         for uid, validation_batch_generator in sampler.get_dataset_batch_generators(batch_size=sampler.batch_size, device=device).items():
#             dataset_losses = []
#             for cont, cat, targets in validation_batch_generator:
#                 val_pred = model(categorical_inputs=cat, continuous_inputs=cont)
#                 # if torch.any(torch.isnan(val_pred)):
#                 #     from IPython import embed; embed(header="string - 63 in train_utils.py ")
#                 loss = loss_fn(val_pred, targets.reshape(-1, 3))

#                 dataset_losses.append(loss)

#                 predictions.append(torch.softmax(val_pred, dim=1).cpu())
#                 truth.append(targets.cpu())
#                 weights.append(torch.full(size=(val_pred.shape[0], 1), fill_value=sampler[uid].relative_weight).cpu())
#             # create event based weight tensor for dataset

#             average_val = sum(dataset_losses) / len(dataset_losses) * sampler[uid].relative_weight
#             val_loss.append(average_val)

#         final_validation_loss = sum(val_loss).cpu()#/ len(val_loss)
#         model.train()

#         truth = torch.concatenate(truth, dim=0)
#         predictions = torch.concatenate(predictions, dim=0)
#         weights = torch.flatten(torch.concatenate(weights, dim=0))
#         return final_validation_loss, (predictions, truth, weights)


@register
def validation_default(model, loss_fn, sampler, device):
    with torch.no_grad():
        # run validation every x steps
        val_loss = []
        model.eval()
        predictions = []
        truth = []
        weights = []

        for uid, validation_batch_generator in sampler.batch_generator(batch_size=sampler.batch_size, device=device).items():
            dataset_losses = []
            for cont, cat, tar in validation_batch_generator:
                cat, cont, tar = cat, cont, tar
                val_pred = model(categorical_inputs=cat, continuous_inputs=cont)
                # if torch.any(torch.isnan(val_pred)):
                #     from IPython import embed; embed(header="string - 63 in train_utils.py ")
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
    # _metrics = metrics.calculate_metrics(
    #     tar,
    #     pred,
    #     label=list(target_map.keys()),
    #     weights=weights,
    # )

    # tensorboard_inst.log_precision(_metrics, step=iteration_step, mode=mode)
    # tensorboard_inst.log_sensitivity(_metrics, step=iteration_step, mode=mode)

training_fn = functions.get(f"training_{train_config.config['training_fn']}")
validation_fn = functions.get(f"validation_{train_config.config['validation_fn']}")
