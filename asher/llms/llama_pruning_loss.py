import torch


def pruning_loss(model, penalty_type="tied_l2_with_lhalf"):
    penalty = 0.0
    for layer in model.model.layers:
        mlp = layer.mlp
        W1 = mlp.gate_proj.weight
        W2 = mlp.up_proj.weight
        W3 = mlp.down_proj.weight

        # Calculate the penalty for the MLP layers
        if penalty_type == "tied_l2_with_lhalf":
            combined = torch.cat([W1, W2, W3.t()], dim=1)
            l2 = torch.norm(combined, p=2, dim=1)
            lhalf_of_l2 = torch.norm(l2, p=0.5, dim=0)
            penalty = penalty + lhalf_of_l2 / (W1.numel() + W2.numel() + W3.numel())

    return penalty
