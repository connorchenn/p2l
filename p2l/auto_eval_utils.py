from typing import Callable, Dict


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import numpy as np
from scipy.optimize import minimize


from scipy.stats import kendalltau, spearmanr
from p2l.model import (
    registered_losses,
    HeadOutputs,
    registered_aggr_models,
    registered_pairwise_losses,
)

registered_simple_metrics: Dict[str, Dict[str, Callable]] = {}
registered_aggr_metrics: Dict[str, Dict[str, Callable]] = {}
registered_helpers: Dict[str, Callable] = {}


def register_simple_metric(loss_type: str, metric: str):
    def decorator(func: Callable):
        if loss_type not in registered_simple_metrics:
            registered_simple_metrics[loss_type] = {}
        registered_simple_metrics[loss_type][metric] = func
        return func

    return decorator


def register_aggr_metric(loss_type: str, metric: str):
    def decorator(func: Callable):
        if loss_type not in registered_aggr_metrics:
            registered_aggr_metrics[loss_type] = {}
        registered_aggr_metrics[loss_type][metric] = func
        return func

    return decorator


def register_helper(loss_or_model_type: str, helper_func):
    def decorator(func: Callable):
        if loss_or_model_type not in registered_helpers:
            registered_helpers[loss_or_model_type] = {}
        registered_helpers[loss_or_model_type][helper_func] = func
        return func

    return decorator


@register_helper("p2l", "output_labels")
def output_labels_p2l(val_data: pd.DataFrame, **kwargs):
    betas = torch.tensor(np.stack(val_data["betas"]), dtype=torch.float)
    labels = torch.tensor(np.stack(val_data["labels"]))
    etas = None

    if "eta" in val_data.columns:
        etas = torch.tensor(np.stack(val_data["eta"]), dtype=torch.float)

    return HeadOutputs(coefs=betas, eta=etas), labels


def translate_coefs(coef, old_list, new_list):
    old_list = old_list.tolist()
    old_to_new = [old_list.index(model) for model in new_list]
    betas_array = np.array(coef)

    betas_array = betas_array[old_to_new]

    return torch.tensor(betas_array)


@register_helper("marginal", "output_labels")
def output_labels_marginal(
    val_data: pd.DataFrame,
    train_data: pd.DataFrame,
    model_list: np.array,
    train_model_list: np.array,
    loss_type: str,
    **kwargs,
):
    train_labels = torch.tensor(np.stack(train_data["labels"]))
    coefs, eta = train_marginal(train_model_list, train_labels, loss_type)
    coefs, eta = coefs[0], eta[0] if eta is not None else None

    coefs = translate_coefs(coefs, train_model_list, model_list)

    val_labels = torch.tensor(np.stack(val_data["labels"]))

    coefs = coefs.expand(len(val_labels), -1)
    eta = eta.expand(len(val_labels), -1) if eta is not None else None

    return HeadOutputs(coefs=coefs, eta=eta), val_labels


@register_helper("marginal-gt", "output_labels")
def output_labels_marginal_gt(
    val_data: pd.DataFrame, model_list: np.array, loss_type: str, **kwargs
):
    val_labels = torch.tensor(np.stack(val_data["labels"]))
    coefs, eta = train_marginal(model_list, val_labels, loss_type)

    coefs = coefs.expand(len(val_labels), -1)
    eta = eta.expand(len(val_labels), -1) if eta is not None else None

    return HeadOutputs(coefs=coefs, eta=eta), val_labels


@register_helper("arena", "output_labels")
def output_labels_arena(
    arena_rankings: torch.tensor, val_data: pd.DataFrame, loss_type: str, **kwargs
):
    labels = torch.tensor(np.stack(val_data["labels"]))

    # arena rankings is already filtered so it will be 1d tensor
    betas = arena_rankings.expand(len(labels), -1)
    etas = torch.ones(len(labels))
    etas = etas.unsqueeze(-1)

    # TODO: Cleanup
    if loss_type == "bt" or loss_type == "bt-tie":
        etas = None

    return HeadOutputs(coefs=betas, eta=etas), labels


@register_helper("bag", "preprocess_data")
def preprocess_data_bag(data: pd.DataFrame, **kwargs):
    condition = data["winner"] == "tie (bothbad)"
    data.loc[condition, "labels"] = data.loc[condition, "labels"].apply(
        lambda arr: arr[:2] + [2]
    )
    return data


@register_helper("bt", "preprocess_data")
@register_helper("bt-tie", "preprocess_data")
@register_helper("rk", "preprocess_data")
@register_helper("rk-reparam", "preprocess_data")
def preprocess_data(data: pd.DataFrame, **kwargs):
    return data


@register_simple_metric("bt", "Loss")
@register_simple_metric("bt", "BCELoss")
@register_simple_metric("bt-tie", "Loss")
@register_simple_metric("rk", "Loss")
@register_simple_metric("rk-reparam", "Loss")
@register_simple_metric("bag", "Loss")
def loss(head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs):
    loss_func = registered_losses.get(loss_type)
    return loss_func(head_output=head_output, labels=labels).item()


@register_simple_metric("rk", "Tie_Loss")
@register_simple_metric("bag", "Tie_Loss")
def tie_loss(head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs):
    loss_func = registered_losses.get("tie-" + loss_type)
    return loss_func(head_output=head_output, labels=labels).item()


@register_simple_metric("bag", "Tie_bb_Loss")
def tie_bb_loss(
    head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs
):
    loss_func = registered_losses.get("tie-bb-" + loss_type)
    return loss_func(head_output=head_output, labels=labels).item()


@register_aggr_metric("bt", "Aggr_Tie_Loss")
@register_aggr_metric("bt-tie", "Aggr_Tie_Loss")
@register_aggr_metric("rk", "Aggr_Tie_Loss")
@register_aggr_metric("rk-reparam", "Aggr_Tie_Loss")
@register_aggr_metric("bag", "Aggr_Tie_Loss")
def Aggr_Tie_Loss(
    gt_output: HeadOutputs,
    model_output: HeadOutputs,
    loss_type: str,
    labels: torch.tensor,
    **kwargs,
):

    return aggr_metric("Tie_Loss", loss_type, labels, gt_output, model_output)


@register_simple_metric("bt-tie", "BCELoss")
@register_simple_metric("rk", "BCELoss")
@register_simple_metric("rk-reparam", "BCELoss")
@register_simple_metric("bag", "BCELoss")
def BCE_loss(head_output: HeadOutputs, labels: torch.Tensor, **kwargs):
    non_tie_index = torch.where(labels[:, -1] == 0)[0]

    new_coefs = head_output.coefs[non_tie_index, :]
    new_eta = head_output.eta[non_tie_index] if head_output.eta is not None else None

    no_tie_output = HeadOutputs(coefs=new_coefs, eta=new_eta)
    no_tie_labels = labels[non_tie_index, :]
    return loss(no_tie_output, no_tie_labels, loss_type="bt")


def aggr_metric(metric_name, loss_type, labels, gt_output, model_output):
    func = registered_simple_metrics[loss_type][metric_name]

    gt = func(
        labels=labels, head_output=expand_output(gt_output, labels), loss_type=loss_type
    )
    model = func(
        labels=labels,
        head_output=expand_output(model_output, labels),
        loss_type=loss_type,
    )

    return {"ground-truth": round(gt, 4), "model-aggr": round(model, 4)}


@register_aggr_metric("bt", "Aggr_Loss")
@register_aggr_metric("bt-tie", "Aggr_Loss")
@register_aggr_metric("rk", "Aggr_Loss")
@register_aggr_metric("rk-reparam", "Aggr_Loss")
@register_aggr_metric("bag", "Aggr_Loss")
def Aggr_Loss(
    gt_output: HeadOutputs,
    model_output: HeadOutputs,
    loss_type: str,
    labels: torch.tensor,
    **kwargs,
):

    return aggr_metric("Loss", loss_type, labels, gt_output, model_output)


@register_aggr_metric("bt", "Aggr_BCELoss")
@register_aggr_metric("bt-tie", "Aggr_BCELoss")
@register_aggr_metric("rk", "Aggr_BCELoss")
@register_aggr_metric("rk-reparam", "Aggr_BCELoss")
@register_aggr_metric("bag", "Aggr_BCELoss")
def Aggr_BCE_Loss(
    gt_output: HeadOutputs,
    model_output: HeadOutputs,
    loss_type: str,
    labels: torch.tensor,
    **kwargs,
):

    return aggr_metric("BCELoss", loss_type, labels, gt_output, model_output)


def expand_output(output, labels):
    coefs, eta = output.coefs, output.eta
    new_coefs = coefs.expand(len(labels), -1)

    if eta is not None:
        eta = eta.expand(len(labels), -1)
    return HeadOutputs(coefs=new_coefs, eta=eta)


@register_simple_metric("bt", "MSELoss")
def BT_mse(
    head_output: HeadOutputs,
    labels: torch.Tensor,
    **kwargs,
):
    coefs = head_output.coefs
    paired_coefs = coefs.gather(dim=-1, index=labels).contiguous()

    paired_delta_logit = paired_coefs[:, 0] - paired_coefs[:, 1]
    predicted_probs = torch.sigmoid(paired_delta_logit)
    true_labels = torch.ones_like(predicted_probs)

    mse = F.mse_loss(predicted_probs, true_labels)
    return mse.mean().item()


@register_simple_metric("bt-tie", "MSELoss")
def BT_tie_mst(
    head_output: HeadOutputs,
    labels: torch.Tensor,
    **kwargs,
):
    coefs = head_output.coefs
    model_idx = labels[:, :2]

    paired_coefs = coefs.gather(dim=-1, index=model_idx).contiguous()
    paired_delta_logit = paired_coefs[:, 0] - paired_coefs[:, 1]

    p_w = torch.sigmoid(paired_delta_logit)
    tie_ind = labels[:, -1]

    # let label be 0.5 if there is tie
    pred_probs = torch.where(tie_ind == 1, 0.5, p_w)

    true_labels = torch.ones_like(pred_probs)
    mse = F.mse_loss(pred_probs, true_labels)
    return mse.mean().item()


@register_simple_metric("rk", "MSELoss")
@register_simple_metric("rk-reparam", "MSELoss")
def RK_mse(head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs):
    probs_func = registered_helpers[loss_type]["probs"]
    p_w, _, p_t = probs_func(head_output=head_output, labels=labels)

    tie_ind = labels[:, -1]

    # True label will always be win (since first index) unless a tie occurs
    pred_probs = torch.where(tie_ind == 1, p_t, p_w)

    true_labels = torch.ones_like(pred_probs)
    mse = F.mse_loss(pred_probs, true_labels)
    return mse.mean().item()


@register_simple_metric("bag", "MSELoss")
def bag_mse(head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs):
    probs_func = registered_helpers[loss_type]["probs"]
    p_w, _, p_t, p_t_bb = probs_func(head_output=head_output, labels=labels)

    tie_ind = labels[:, -1].unsqueeze(-1)

    P = torch.stack([p_w, p_t, p_t_bb], dim=-1)

    pred_probs = P.gather(dim=-1, index=tie_ind).contiguous().squeeze(-1)

    true_labels = torch.ones_like(pred_probs)
    mse = F.mse_loss(pred_probs, true_labels)
    return mse.mean().item()


@register_helper("rk-reparam", "probs")
def rk_reparam_probs(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):
    coefs = head_output.coefs
    eta = head_output.eta

    theta = (torch.exp(eta) + 1.000001).squeeze(-1)

    winner_idx = labels[:, 0:1]
    loser_idx = labels[:, 1:2]

    beta_win = coefs.gather(dim=-1, index=winner_idx).contiguous()[:, 0]
    beta_lose = coefs.gather(dim=-1, index=loser_idx).contiguous()[:, 0]

    pi_win = torch.exp(beta_win)
    pi_lose = torch.exp(beta_lose)
    p_win = pi_win / (pi_win + theta * pi_lose + 1.0)

    p_lose = pi_lose / (pi_lose + theta * pi_win + 1.0)

    p_tie = 1.0 - p_win - p_lose
    return p_win, p_lose, p_tie


@register_helper("bag", "probs")
def bag_probs(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):
    coefs = head_output.coefs
    eta = head_output.eta

    theta = (torch.exp(eta) + 1.000001).squeeze(-1)

    winner_idx = labels[:, 0:1]
    loser_idx = labels[:, 1:2]

    beta_win = coefs.gather(dim=-1, index=winner_idx).contiguous()[:, 0]
    beta_lose = coefs.gather(dim=-1, index=loser_idx).contiguous()[:, 0]

    pi_win = torch.exp(beta_win)
    pi_lose = torch.exp(beta_lose)
    pi_gamma = 1.0

    p_win = pi_win / (pi_win + theta * pi_lose + pi_gamma)
    p_lose = pi_lose / (pi_lose + theta * pi_win + pi_gamma)
    p_tie_bb = pi_gamma / (pi_gamma + pi_win + pi_lose)

    p_tie = 1.0 - p_win - p_lose - p_tie_bb
    return p_win, p_lose, p_tie, p_tie_bb


@register_helper("rk", "probs")
def rk_probs(
    head_output: HeadOutputs, labels: Dict, weights: torch.Tensor = None, **kwargs
):
    coefs = head_output.coefs

    eta = rk_eta(head_output)

    model_idx = labels[:, :2]
    paired_coefs = coefs.gather(dim=-1, index=model_idx).contiguous()
    paired_delta_logit = paired_coefs[:, 0] - paired_coefs[:, 1]

    p_w = torch.sigmoid(paired_delta_logit - eta)
    p_l = torch.sigmoid(-1 * paired_delta_logit - eta)
    p_t = 1 - p_w - p_l

    return p_w, p_l, p_t


@register_simple_metric("bt", "Accuracy")
def BT_accuracy(
    head_output: HeadOutputs,
    labels: torch.Tensor,
    **kwargs,
):
    coefs = head_output.coefs
    paired_coefs = coefs.gather(dim=-1, index=labels).contiguous()
    paired_delta_logit = paired_coefs[:, 0] - paired_coefs[:, 1]

    # winner would have positive difference
    correct = (paired_delta_logit > 0).float()
    return correct.mean().item()


@register_simple_metric("bt-tie", "Accuracy")
def BT_tie_accuracy(
    head_output: HeadOutputs,
    labels: torch.Tensor,
    **kwargs,
):
    coefs = head_output.coefs
    paired_coefs = coefs.gather(dim=-1, index=labels).contiguous()

    paired_delta_logit = paired_coefs[:, 0] - paired_coefs[:, 1]

    # winner would have positive difference
    correct = (paired_delta_logit > 0).float()
    tie_ind = labels[:, -1]
    # we give ties half the accuracy
    correct[tie_ind == 1] = 0.5
    return correct.mean().item()


@register_simple_metric("rk", "Accuracy")
@register_simple_metric("rk-reparam", "Accuracy")
def RK_accuracy(
    head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs
):
    probs_func = registered_helpers[loss_type]["probs"]
    p_w, p_l, p_t = probs_func(head_output=head_output, labels=labels)

    pred_labels = torch.where(
        p_w >= p_l, torch.where(p_w >= p_t, 1, 0.5), torch.where(p_l >= p_t, 0, 0.5)
    )

    tie_ind = labels[:, -1]
    # tie if tie index, else winner (first index) predicted to win
    true_labels = torch.where(tie_ind == 1, 0.5, 1)

    correct = (pred_labels == true_labels).float()
    return correct.mean().item()


@register_simple_metric("rk", "Tie_Accuracy")
def RK_tie_accuracy(
    head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs
):
    probs_func = registered_helpers[loss_type]["probs"]
    p_w, p_l, p_t = probs_func(head_output=head_output, labels=labels)

    p_nt = p_w + p_l

    pred_tie = torch.where(p_t >= p_nt, 1, 0)

    tie_ind = labels[:, -1]
    correct = (pred_tie == tie_ind).float()
    return correct.mean().item()


@register_simple_metric("bag", "Tie_Accuracy")
def bag_tie_accuracy(
    head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs
):
    probs_func = registered_helpers[loss_type]["probs"]
    p_w, p_l, p_t, p_t_bb = probs_func(head_output=head_output, labels=labels)

    p_nt = p_w + p_l
    p_tie = p_t + p_t_bb

    pred_tie = torch.where(p_nt >= p_tie, 0, 1)

    tie_ind = torch.where(labels[:, -1] == 0, 0, 1)
    correct = (pred_tie == tie_ind).float()
    return correct.mean().item()


@register_simple_metric("bag", "Tie_bb_Accuracy")
def bag_tie_bb_accuracy(
    head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs
):
    probs_func = registered_helpers[loss_type]["probs"]
    p_w, p_l, p_t, p_t_bb = probs_func(head_output=head_output, labels=labels)

    p_nt_bb = p_w + p_l + p_t

    pred_tie = torch.where(p_t_bb >= p_nt_bb, 1, 0)

    tie_ind = torch.where(labels[:, -1] == 2, 1, 0)
    correct = (pred_tie == tie_ind).float()
    return correct.mean().item()


@register_aggr_metric("bt", "Aggr_Tie_Accuracy")
@register_aggr_metric("bt-tie", "Aggr_Tie_Accuracy")
@register_aggr_metric("rk", "Aggr_Tie_Accuracy")
@register_aggr_metric("rk-reparam", "Aggr_Tie_Accuracy")
@register_aggr_metric("bag", "Aggr_Tie_Accuracy")
def Aggr_Tie_accuracy(
    gt_output: HeadOutputs,
    model_output: HeadOutputs,
    loss_type: str,
    labels: torch.tensor,
    **kwargs,
):

    return aggr_metric("Tie_Accuracy", loss_type, labels, gt_output, model_output)


@register_aggr_metric("bt", "Aggr_Tie_Accuracy")
@register_aggr_metric("bt-tie", "Aggr_Tie_Accuracy")
@register_aggr_metric("rk", "Aggr_Tie_Accuracy")
@register_aggr_metric("rk-reparam", "Aggr_Tie_Accuracy")
@register_aggr_metric("bag", "Aggr_Tie_Accuracy")
def Aggr_Tie_accuracy(
    gt_output: HeadOutputs,
    model_output: HeadOutputs,
    loss_type: str,
    labels: torch.tensor,
    **kwargs,
):

    return aggr_metric("Tie_Accuracy", loss_type, labels, gt_output, model_output)


@register_aggr_metric("bt", "Aggr_Tie_bb_Accuracy")
@register_aggr_metric("bt-tie", "Aggr_Tie_bb_Accuracy")
@register_aggr_metric("rk", "Aggr_Tie_bb_Accuracy")
@register_aggr_metric("rk-reparam", "Aggr_Tie_bb_Accuracy")
@register_aggr_metric("bag", "Aggr_Tie_bb_Accuracy")
def Aggr_Tie_bb_accuracy(
    gt_output: HeadOutputs,
    model_output: HeadOutputs,
    loss_type: str,
    labels: torch.tensor,
    **kwargs,
):

    return aggr_metric("Tie_bb_Accuracy", loss_type, labels, gt_output, model_output)


@register_aggr_metric("bt", "Aggr_Tie_bb_Loss")
@register_aggr_metric("bt-tie", "Aggr_Tie_bb_Loss")
@register_aggr_metric("rk", "Aggr_Tie_bb_Loss")
@register_aggr_metric("rk-reparam", "Aggr_Tie_bb_Loss")
@register_aggr_metric("bag", "Aggr_Tie_bb_Loss")
def Aggr_Tie_bb_loss(
    gt_output: HeadOutputs,
    model_output: HeadOutputs,
    loss_type: str,
    labels: torch.tensor,
    **kwargs,
):

    return aggr_metric("Tie_bb_Loss", loss_type, labels, gt_output, model_output)


@register_simple_metric("rk-reparam", "Tie_Accuracy")
@register_simple_metric("bt", "Tie_Accuracy")
@register_simple_metric("bt-tie", "Tie_Accuracy")
@register_simple_metric("bt", "Tie_bb_Loss")
@register_simple_metric("rk-reparam", "Tie_bb_Loss")
@register_simple_metric("bt-tie", "Tie_bb_Loss")
@register_simple_metric("rk", "Tie_bb_Loss")
@register_simple_metric("bt", "Tie_Loss")
@register_simple_metric("bt-tie", "Tie_Loss")
@register_simple_metric("rk-reparam", "Tie_Loss")
@register_simple_metric("rk", "Tie_bb_Accuracy")
@register_simple_metric("rk-reparam", "Tie_bb_Accuracy")
@register_simple_metric("bt", "Tie_bb_Accuracy")
@register_simple_metric("bt-tie", "Tie_bb_Accuracy")
def not_implemented(
    head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs
):
    return -1  # not implemented


@register_simple_metric("bag", "Accuracy")
def bag_accuracy(
    head_output: HeadOutputs, labels: torch.Tensor, loss_type: str, **kwargs
):
    probs_func = registered_helpers[loss_type]["probs"]
    p_w, p_l, p_t, p_t_bb = probs_func(head_output=head_output, labels=labels)

    P = torch.stack([p_w, p_t, p_t_bb, p_l], dim=-1)

    pred_labels = P.argmax(dim=-1)

    tie_ind = labels[:, -1]
    # let win be 0, tie be 1, tie_bb be 2. loss never predicted since winner_idx first
    true_labels = tie_ind

    correct = (pred_labels == true_labels).float()
    return correct.mean().item()


@register_simple_metric("bt", "Mean-BT")
@register_simple_metric("bt-tie", "Mean-BT")
@register_simple_metric("rk", "Mean-BT")
@register_simple_metric("rk-reparam", "Mean-BT")
@register_simple_metric("bag", "Mean-BT")
def beta_mean(
    head_output: HeadOutputs,
    **kwargs,
):
    betas = head_output.coefs
    flat_betas = betas.flatten()
    return torch.mean(flat_betas).item()


@register_simple_metric("bt", "Std-BT")
@register_simple_metric("bt-tie", "Std-BT")
@register_simple_metric("rk", "Std-BT")
@register_simple_metric("rk-reparam", "Std-BT")
@register_simple_metric("bag", "Std-BT")
def beta_std(
    head_output: HeadOutputs,
    **kwargs,
):
    betas = head_output.coefs
    flat_betas = betas.flatten()
    return torch.std(flat_betas).item()


@register_simple_metric("bt", "Spread-BT")
@register_simple_metric("bt-tie", "Spread-BT")
@register_simple_metric("rk", "Spread-BT")
@register_simple_metric("rk-reparam", "Spread-BT")
@register_simple_metric("bag", "Spread-BT")
def beta_spread(
    head_output: HeadOutputs,
    **kwargs,
):
    betas = head_output.coefs
    flat_betas = betas.flatten()
    return (torch.max(flat_betas) - torch.min(flat_betas)).item()


@register_simple_metric("bt", "Mean-Spread-BT")
@register_simple_metric("bt-tie", "Mean-Spread-BT")
@register_simple_metric("rk", "Mean-Spread-BT")
@register_simple_metric("rk-reparam", "Mean-Spread-BT")
@register_simple_metric("bag", "Mean-Spread-BT")
def beta_mean_spread(
    head_output: HeadOutputs,
    **kwargs,
):
    betas = head_output.coefs
    max_min_per_prompt = (
        torch.max(betas, dim=-1).values - torch.min(betas, dim=-1).values
    )
    return torch.mean(max_min_per_prompt).item()


@register_simple_metric("bt", "Mean-IQR-BT")
@register_simple_metric("bt-tie", "Mean-IQR-BT")
@register_simple_metric("rk", "Mean-IQR-BT")
@register_simple_metric("rk-reparam", "Mean-IQR-BT")
@register_simple_metric("bag", "Mean-IQR-BT")
def beta_mean_iqr(
    head_output: HeadOutputs,
    **kwargs,
):
    betas = head_output.coefs
    iqr_per_prompt = torch.quantile(betas, 0.75, dim=-1) - torch.quantile(
        betas, 0.25, dim=-1
    )
    return torch.mean(iqr_per_prompt).item()


@register_simple_metric("bt", "Mean-Std-BT")
@register_simple_metric("bt-tie", "Mean-Std-BT")
@register_simple_metric("rk", "Mean-Std-BT")
@register_simple_metric("rk-reparam", "Mean-Std-BT")
@register_simple_metric("bag", "Mean-Std-BT")
def beta_mean_std(
    head_output: HeadOutputs,
    **kwargs,
):
    betas = head_output.coefs
    std_per_prompt = torch.std(betas, dim=-1)
    return torch.mean(std_per_prompt).item()


@register_helper("marginal-gt", "aggregrate")
def aggr_marginal_gt(
    labels: torch.Tensor, model_list: torch.Tensor, loss_type: str, **kwargs
):
    coefs, eta = train_marginal(model_list, labels, loss_type)
    return HeadOutputs(coefs=coefs[0], eta=eta[0] if eta is not None else None)


@register_helper("p2l", "aggregrate")
def aggr_p2l(
    head_output: HeadOutputs,
    labels: torch.Tensor,
    model_list: torch.Tensor,
    loss_type: str,
    **kwargs,
):
    coefs, eta = train_aggr_prob(
        model_list, head_output, labels, loss_type, is_batch=False
    )
    return HeadOutputs(coefs=coefs[0], eta=eta[0] if eta is not None else None)


@register_helper("p2l", "aggregrate-batch")
def aggr_p2l_batch(
    head_output: HeadOutputs,
    labels: torch.Tensor,
    model_list: torch.Tensor,
    loss_type: str,
    **kwargs,
):
    coefs_batch, eta_batch = train_aggr_prob(
        model_list, head_output, labels, loss_type, is_batch=True
    )
    return [
        HeadOutputs(
            coefs=coefs_batch[i], eta=eta_batch[i] if eta_batch is not None else None
        )
        for i in range(coefs_batch.shape[0])
    ]


@register_helper("marginal-gt", "aggregrate-batch")
def aggr_p2l_batch(
    head_output: HeadOutputs,
    labels: torch.Tensor,
    model_list: torch.Tensor,
    loss_type: str,
    **kwargs,
):
    # TODO: Make faster if necessary
    return [
        aggr_marginal_gt(labels[i], model_list, loss_type) for i in range(len(labels))
    ]


@register_helper("marginal", "aggregrate")
def aggr_non_p2l(head_output: HeadOutputs, loss_type: str, **kwargs):
    etas = head_output.eta
    etas = etas[0, :] if etas is not None else None
    return HeadOutputs(coefs=head_output.coefs[0, :], eta=etas)


@register_helper("arena", "aggregrate")
def aggr_non_p2l(
    head_output: HeadOutputs = None, arena_rankings: torch.tensor = None, **kwargs
):
    eta = torch.tensor([0])

    if arena_rankings is not None:
        return HeadOutputs(coefs=arena_rankings, eta=eta)
    # arena just has the same betas repeated if not provided
    return HeadOutputs(coefs=head_output.coefs[0, :], eta=eta)


def train_marginal(model_list, labels, loss_type, lr=1.0, tol=1e-9, max_epochs=50):
    model_cls = registered_aggr_models[loss_type]
    model = model_cls(len(model_list))

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_epochs,
        tolerance_grad=tol,
        tolerance_change=tol,
    )

    loss_func = registered_losses[loss_type]
    labels = (
        labels.squeeze() if labels.dim() > 2 else labels
    )  # marginal doesn't use batching since one at a time

    def closure():
        optimizer.zero_grad()
        coefs, eta = model()

        coefs_expanded = coefs[0].expand(len(labels), -1)
        eta_expanded = eta[0].expand(len(labels), -1) if eta is not None else None

        head_output = HeadOutputs(coefs=coefs_expanded, eta=eta_expanded)
        loss = loss_func(head_output=head_output, labels=labels)
        loss.backward()
        return loss

    optimizer.step(closure)

    true_coefs, true_eta = model()
    return true_coefs.detach(), true_eta.detach() if true_eta is not None else None


def train_aggr_prob(
    model_list,
    head_outputs,
    labels,
    loss_type,
    is_batch,
    lr=1.0,
    tol=1e-9,
    max_epochs=50,
):
    true_probs_func = registered_helpers[loss_type]["pairwise_probs"]
    true_probs = true_probs_func(real_output=head_outputs)
    # add a batch size of 1 since aggregration is done in batches (only necessary if data isn't in batch format)
    if not is_batch:
        true_probs = true_probs.unsqueeze(0)

    batch_size = true_probs.shape[0]
    model_cls = registered_aggr_models[loss_type]
    model = model_cls(len(model_list), batch_size)

    optimizer = optim.LBFGS(
        model.parameters(),
        lr=lr,
        max_iter=max_epochs,
        tolerance_grad=tol,
        tolerance_change=tol,
    )
    loss_func = registered_pairwise_losses[loss_type]

    count = 0
    prev_loss = 0

    def closure():
        optimizer.zero_grad()
        coefs, eta = model()
        aggr_output = HeadOutputs(coefs=coefs, eta=eta)
        loss = loss_func(
            real_output=head_outputs,
            aggregated_output=aggr_output,
            true_probs=true_probs,
        )
        loss.backward()

        nonlocal count
        count += 1
        if count == 49:
            raise Warning("Batch training did not converge")

        return loss

    optimizer.step(closure)

    true_coefs, true_eta = model()
    return true_coefs.detach(), true_eta.detach() if true_eta is not None else None


def rk_eta(output):
    if output.eta is None:
        return None
    BETA = 0.1
    return torch.clamp(
        torch.nn.functional.softplus(output.eta - 22.5, BETA).squeeze(-1), min=0.02
    )


@register_helper("rk", "pairwise_probs")
def pairwise_RK_probs(real_output: HeadOutputs):

    real_betas = real_output.coefs
    real_eta = rk_eta(real_output)
    real_eta = real_eta.unsqueeze(-1)

    num_models = real_betas.shape[-1]

    pair_indices = torch.tensor(
        [(i, j) for i in range(num_models) for j in range(i + 1, num_models)],
        dtype=torch.long,
    )

    # elipses allow for both batched/unbatched
    beta_i_real = real_betas[..., pair_indices[:, 0]]
    beta_j_real = real_betas[..., pair_indices[:, 1]]

    true_probs_win = torch.sigmoid(beta_i_real - beta_j_real - real_eta)
    true_probs_loss = torch.sigmoid(beta_j_real - beta_i_real - real_eta)
    true_probs_tie = 1.0 - true_probs_win - true_probs_loss

    true_probs = torch.stack((true_probs_win, true_probs_loss, true_probs_tie), dim=-1)
    return true_probs


@register_helper("rk-reparam", "pairwise_probs")
def pairwise_RK_reparam_probs(real_output: HeadOutputs, **kwargs):
    real_betas = real_output.coefs
    real_theta = torch.exp(real_output.eta) + 1.000001

    num_models = real_betas.shape[-1]

    pair_indices = torch.tensor(
        [(i, j) for i in range(num_models) for j in range(i + 1, num_models)],
        dtype=torch.long,
    )

    beta_i_real = real_betas[..., pair_indices[:, 0]]
    beta_j_real = real_betas[..., pair_indices[:, 1]]

    pi_win = torch.exp(beta_i_real)
    pi_lose = torch.exp(beta_j_real)

    p_win = pi_win / (pi_win + real_theta * pi_lose + 1.0)
    p_lose = pi_lose / (pi_lose + real_theta * pi_win + 1.0)
    p_tie = 1.0 - p_win - p_lose

    true_probs = torch.stack((p_win, p_lose, p_tie), dim=-1)
    return true_probs


@register_helper("bag", "pairwise_probs")
def pairwise_bag_probs(real_output: HeadOutputs, **kwargs):
    real_betas = real_output.coefs
    real_theta = torch.exp(real_output.eta) + 1.000001

    num_models = real_betas.shape[-1]

    pair_indices = torch.tensor(
        [(i, j) for i in range(num_models) for j in range(i + 1, num_models)],
        dtype=torch.long,
    )

    beta_i_real = real_betas[..., pair_indices[:, 0]]
    beta_j_real = real_betas[..., pair_indices[:, 1]]

    pi_win = torch.exp(beta_i_real)
    pi_lose = torch.exp(beta_j_real)
    pi_gamma = 1.0

    p_win = pi_win / (pi_win + real_theta * pi_lose + pi_gamma)

    p_lose = pi_lose / (pi_lose + real_theta * pi_win + pi_gamma)

    p_tie_bb = pi_gamma / (pi_gamma + pi_win + pi_lose)

    p_tie = 1.0 - p_win - p_lose - p_tie_bb

    true_probs = torch.stack((p_win, p_lose, p_tie, p_tie_bb), dim=-1)
    return true_probs


@register_helper("bt", "pairwise_probs")
@register_helper("bt-tie", "pairwise_probs")
def pairwise_BT_probs(real_output: HeadOutputs):
    real_betas = real_output.coefs

    num_models = real_betas.shape[-1]

    pair_indices = torch.tensor(
        [(i, j) for i in range(num_models) for j in range(i + 1, num_models)],
        dtype=torch.long,
    )

    beta_i_real = real_betas[..., pair_indices[:, 0]]
    beta_j_real = real_betas[..., pair_indices[:, 1]]

    true_probs = torch.sigmoid(beta_i_real - beta_j_real)
    return true_probs


# removes nan from tensor, indices will be shifted
def remove_beta_nan(beta1, beta2):
    beta_mask = ~torch.isnan(beta1) & ~torch.isnan(beta2)
    return beta1[beta_mask], beta2[beta_mask]


@register_aggr_metric("bt", "Leaderboard")
@register_aggr_metric("bt-tie", "Leaderboard")
@register_aggr_metric("rk", "Leaderboard")
@register_aggr_metric("rk-reparam", "Leaderboard")
@register_aggr_metric("bag", "Leaderboard")
def leaderboard(
    gt_output: HeadOutputs, model_output: HeadOutputs, model_list: np.array, **kwargs
):
    gt_lb = get_leaderboard(gt_output, model_list)
    model_lb = get_leaderboard(model_output, model_list)

    return {"ground-truth": list(gt_lb), "model-aggr": list(model_lb)}


def get_leaderboard(output, model_list):
    coefs = output.coefs

    sorted_indices = torch.argsort(coefs, descending=True)
    sorted_model_names = [model_list[i] for i in sorted_indices]
    sorted_betas = coefs[sorted_indices]

    leaderboard = []
    for i in range(len(sorted_model_names)):
        beta = (
            round(sorted_betas[i].item(), 4)
            if not torch.isnan(sorted_betas[i])
            else "nan"
        )
        cur_model = str(sorted_model_names[i]) + ": " + str(beta)
        leaderboard.append(cur_model)

    return np.array(leaderboard)


@register_aggr_metric("bt", "L1-Dist-Prob")
@register_aggr_metric("bt-tie", "L1-Dist-Prob")
def l1_dist_prob_bt(gt_output: HeadOutputs, model_output: HeadOutputs, **kwargs):
    beta1 = gt_output.coefs
    beta2 = model_output.coefs

    # if arena is one, there may be nan if model not present in that file
    beta1, beta2 = remove_beta_nan(beta1, beta2)

    diff_matrix1 = beta1.unsqueeze(1) - beta1.unsqueeze(0)
    diff_matrix2 = beta2.unsqueeze(1) - beta2.unsqueeze(0)

    prob_vec1 = torch.sigmoid(diff_matrix1).flatten()
    prob_vec2 = torch.sigmoid(diff_matrix2).flatten()

    return torch.abs(prob_vec2 - prob_vec1).mean().item()


@register_aggr_metric("rk-reparam", "L1-Dist-Prob")
@register_aggr_metric("rk", "L1-Dist-Prob")
def l1_dist_prob_rk(
    gt_output: HeadOutputs, model_output: HeadOutputs, loss_type: str, **kwargs
):
    eta1 = gt_output.eta
    eta2 = model_output.eta
    # need to both have eta
    if eta1 is None or eta2 is None:
        return l1_dist_prob_bt(gt_output, model_output)

    pair_probs_func = registered_helpers[loss_type]["pairwise_probs"]

    p_win1, p_lose1, p_tie1 = torch.unbind(pair_probs_func(gt_output), -1)
    p_win2, p_lose2, p_tie2 = torch.unbind(pair_probs_func(model_output), -1)

    win_diff = torch.abs(p_win1 - p_win2).mean().item()
    lose_diff = torch.abs(p_lose1 - p_lose2).mean().item()
    tie_diff = torch.abs(p_tie1 - p_tie2).mean().item()
    return (win_diff + lose_diff + tie_diff) / 3


@register_aggr_metric("bag", "L1-Dist-Prob")
def l1_dist_prob_bag(
    gt_output: HeadOutputs, model_output: HeadOutputs, loss_type: str, **kwargs
):
    eta1 = gt_output.eta
    eta2 = model_output.eta
    # need to both have eta
    if eta1 is None or eta2 is None:
        return l1_dist_prob_bt(gt_output, model_output)

    pair_probs_func = registered_helpers[loss_type]["pairwise_probs"]

    p_win1, p_lose1, p_tie1, p_tie_bb1 = torch.unbind(pair_probs_func(gt_output), -1)
    p_win2, p_lose2, p_tie2, p_tie_bb2 = torch.unbind(pair_probs_func(model_output), -1)

    win_diff = torch.abs(p_win1 - p_win2).mean().item()
    lose_diff = torch.abs(p_lose1 - p_lose2).mean().item()
    tie_diff = torch.abs(p_tie1 - p_tie2).mean().item()
    tie_bb_diff = torch.abs(p_tie_bb2 - p_tie_bb1).mean().item()
    return (win_diff + lose_diff + tie_diff + tie_bb_diff) / 4


@register_aggr_metric("bt", "IQR-BT")
@register_aggr_metric("bt-tie", "IQR-BT")
@register_aggr_metric("rk", "IQR-BT")
@register_aggr_metric("rk-reparam", "IQR-BT")
@register_aggr_metric("bag", "IQR-BT")
def beta_iqr(gt_output: HeadOutputs, model_output: HeadOutputs, **kwargs):
    (
        gt_coefs,
        model_coefs,
    ) = (
        gt_output.coefs,
        model_output.coefs,
    )
    gt_iqr = (torch.quantile(gt_coefs, 0.75) - torch.quantile(gt_coefs, 0.25)).item()
    model_iqr = (
        torch.quantile(model_coefs, 0.75) - torch.quantile(model_coefs, 0.25)
    ).item()

    return {"ground-truth": round(gt_iqr, 4), "model-aggr": round(model_iqr, 4)}


@register_aggr_metric("bt", "Std-BT")
@register_aggr_metric("bt-tie", "Std-BT")
@register_aggr_metric("rk", "Std-BT")
@register_aggr_metric("rk-reparam", "Std-BT")
@register_aggr_metric("bag", "Std-BT")
def beta_std_aggr(gt_output: HeadOutputs, model_output: HeadOutputs, **kwargs):

    gt_betas, model_betas = gt_output.coefs, model_output.coefs
    gt_std, model_std = (
        torch.std(gt_betas.flatten()).item(),
        torch.std(model_betas.flatten()).item(),
    )
    return {"ground-truth": round(gt_std, 4), "model-aggr": round(model_std, 4)}


@register_aggr_metric("bt", "Spread-BT")
@register_aggr_metric("bt-tie", "Spread-BT")
@register_aggr_metric("rk", "Spread-BT")
@register_aggr_metric("rk-reparam", "Spread-BT")
@register_aggr_metric("bag", "Spread-BT")
def beta_spread_aggr(gt_output: HeadOutputs, model_output: HeadOutputs, **kwargs):
    gt_betas, model_betas = gt_output.coefs.flatten(), model_output.coefs.flatten()

    gt_spread, model_spread = torch.max(gt_betas) - torch.min(gt_betas), torch.max(
        model_betas
    ) - torch.min(model_betas)
    return {
        "ground-truth": round(gt_spread.item(), 4),
        "model-aggr": round(model_spread.item(), 4),
    }


@register_aggr_metric("bt", "Kendall-lbs")
@register_aggr_metric("bt-tie", "Kendall-lbs")
@register_aggr_metric("rk", "Kendall-lbs")
@register_aggr_metric("rk-reparam", "Kendall-lbs")
@register_aggr_metric("bag", "Kendall-lbs")
def kendall_lb(gt_output: HeadOutputs, model_output: HeadOutputs, **kwargs):
    gt_betas, model_betas = remove_beta_nan(gt_output.coefs, model_output.coefs)
    gt_lb = gt_betas.numpy()
    model_lb = model_betas.numpy()

    return kendalltau(gt_lb, model_lb)[0]


@register_aggr_metric("bt", "Spearman-lbs")
@register_aggr_metric("bt-tie", "Spearman-lbs")
@register_aggr_metric("rk", "Spearman-lbs")
@register_aggr_metric("rk-reparam", "Spearman-lbs")
@register_aggr_metric("bag", "Spearman-lbs")
def spearman_lb(gt_output: HeadOutputs, model_output: HeadOutputs, **kwargs):
    gt_betas, model_betas = remove_beta_nan(gt_output.coefs, model_output.coefs)
    gt_lb = gt_betas.numpy()
    model_lb = model_betas.numpy()

    return spearmanr(gt_lb, model_lb)[0]


def top_k_frac(gt_betas: torch.tensor, model_betas: torch.tensor, k: int):
    gt_top_indices = set(torch.topk(gt_betas, k).indices.numpy())
    model_top_indices = set(torch.topk(model_betas, k).indices.numpy())
    common_indices = gt_top_indices & model_top_indices

    return len(common_indices) / k


def top_k_displace(gt_betas: torch.tensor, model_betas: torch.tensor, k: int):
    gt_top_indices = torch.topk(gt_betas, k).indices
    model_ranks = torch.argsort(torch.argsort(model_betas, descending=True))
    displacements = torch.abs(model_ranks[gt_top_indices] - torch.arange(k))

    return displacements.float().mean().item()


@register_aggr_metric("bt", "Top-k-fraction")
@register_aggr_metric("bt-tie", "Top-k-fraction")
@register_aggr_metric("rk", "Top-k-fraction")
@register_aggr_metric("rk-reparam", "Top-k-fraction")
@register_aggr_metric("bag", "Top-k-fraction")
def top_k_frac_dict(gt_output: HeadOutputs, model_output: HeadOutputs, **kwargs):
    gt_betas, model_betas = remove_beta_nan(gt_output.coefs, model_output.coefs)

    res = {}
    for k in [1, 3, 5, 10]:
        res[k] = round(top_k_frac(gt_betas, model_betas, k), 4)

    return res


@register_aggr_metric("bt", "Top-k-displace")
@register_aggr_metric("bt-tie", "Top-k-displace")
@register_aggr_metric("rk", "Top-k-displace")
@register_aggr_metric("rk-reparam", "Top-k-displace")
@register_aggr_metric("bag", "Top-k-displace")
def top_k_dist_dict(gt_output: HeadOutputs, model_output: HeadOutputs, **kwargs):
    gt_betas, model_betas = remove_beta_nan(gt_output.coefs, model_output.coefs)

    res = {}
    for k in [1, 3, 5, 10]:
        res[k] = round(top_k_displace(gt_betas, model_betas, k), 4)

    return res
