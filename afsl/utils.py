import torch
from afsl.types import Model


def get_device(model: Model):
    return next(model.parameters()).device


def mini_batch_wrapper_non_cat(fn, data, batch_size):
    n_batches = int(data.size(0) / batch_size)
    results = []
    for i in range(n_batches + 1):
        mini_batch = data[i * batch_size : (i + 1) * batch_size]
        if len(mini_batch) == 0:
            continue
        result = fn(mini_batch)
        results.append(result)
    return results


def mini_batch_wrapper(fn, data, batch_size):
    results = mini_batch_wrapper(fn, data, batch_size)
    return torch.cat(results, dim=0)
