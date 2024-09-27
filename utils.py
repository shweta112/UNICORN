import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler


def replace_none_with_false(d):
    for key, value in d.items():
        if value is None:
            d[key] = False
    return d

def get_loss_weighting(label_array):
    val,count=np.unique(label_array,return_counts=True)
    return np.average(count)/count #average loss weighting is 1

def calculate_stats(dicts):
    keys = list(dicts[0].keys())
    values = [list(d.values()) for d in dicts]
    flat_values = np.concatenate(values)
    avg = np.mean(flat_values)
    std_dev = np.std(flat_values)
    result = {}
    for i, key in enumerate(keys):
        result[key] = {'average': np.mean([v[i] for v in values]),
                       'standard deviation': np.std([v[i] for v in values])}
    return result

def get_loss(name, **kwargs):
    # Check if the name is a valid loss name
    if name == 'BCEWithLogitsLoss':
        kwargs['pos_weight']=kwargs['weight'][1]/kwargs['weight'][0]
        del kwargs['weight']
    if name in nn.__dict__:
        # Get the loss class from the torch.nn module
        loss_class = getattr(nn, name)
        # Instantiate the loss with the reduction option
        loss = loss_class(**kwargs)
        # Return the loss
        return loss
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid loss name: {name}")


def get_optimizer(name, model, lr=0.01, wd=0.1):
    # Check if the name is a valid optimizer name
    if name in optim.__dict__:
        # Get the optimizer class from the torch.optim module
        optimizer_class = getattr(optim, name)
        # Instantiate the optimizer with the model parameters and the learning rate
        optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=wd)
        # Return the optimizer
        return optimizer
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid optimizer name: {name}")


def get_scheduler(name, optimizer, **kwargs):
    # Check if the name is a valid scheduler name
    if name in lr_scheduler.__dict__:
        # Get the scheduler class from the torch.optim.lr_scheduler module
        scheduler_class = getattr(lr_scheduler, name)
        # Instantiate the scheduler with the optimizer and other keyword arguments
        scheduler = scheduler_class(optimizer, **kwargs)
        # Return the scheduler
        return scheduler
    else:
        # Raise an exception if the name is not valid
        raise ValueError(f"Invalid scheduler name: {name}")
    
