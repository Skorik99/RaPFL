import torch
import numpy as np


def get_loss(loss_cfg, df, device, num_classes):
    loss_name = loss_cfg.loss_name
    loss = loss_cfg.config
    init_pos_weight = loss_cfg.config.init_pos_weight
    if loss_name == "ce":
        if init_pos_weight:
            pos_weight = calc_pos_weight_multi_class(df, num_classes)
            pos_weight = torch.tensor(np.array(pos_weight)).to(device)
        elif loss.pos_weight is not None:
            pos_weight = loss.pos_weight
            pos_weight = torch.tensor(np.array(pos_weight)).to(device)
        else:
            pos_weight = None

        return torch.nn.CrossEntropyLoss(
            weight=pos_weight,
            ignore_index=loss.ignore_index,
            reduction=loss.reduction,
            label_smoothing=loss.label_smoothing,
        )
    else:
        raise ValueError("Unknown type of loss function")


def calc_pos_weight_multi_class(df, num_classes):
    targets = [
        t[0] if isinstance(t, (list, tuple, np.ndarray)) else t for t in df["target"]
    ]
    targets = np.array(targets, dtype=np.int64)

    class_counts = np.bincount(targets, minlength=num_classes)

    total = len(targets)

    weights = np.zeros(num_classes, dtype=np.float32)
    for i in range(num_classes):
        if class_counts[i] > 0:
            weights[i] = total / (num_classes * class_counts[i])
        else:
            # if the class does not occur at all, we set any number (for example, 1)
            weights[i] = 1.0

    return weights
