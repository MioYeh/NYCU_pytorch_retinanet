"""Utilities for determining how ImageNet normalisation is applied to a model."""


def resolve_imagenet_norm_mode(net):
    """Return the imagenet normalisation mode for *net*.

    Returns
    -------
    str
        ``'dataloader'``  – normalisation is applied externally (in the eval
        script's ``preprocess()`` function) before feeding the tensor to the
        model.  This is the standard behaviour for all checkpoints that do
        **not** contain an ``input_preprocessor`` with built-in normalisation.

        ``'model'``  – the model contains an ``input_preprocessor`` that
        applies normalisation internally, so the eval script should skip it.
    """
    import torch.nn as nn
    module = net.module if isinstance(net, nn.DataParallel) else net
    input_preprocessor = getattr(module, 'input_preprocessor', None)
    if input_preprocessor is None:
        return 'dataloader'
    # If there is an input_preprocessor but no amp_norm, still use dataloader
    # normalisation (the preprocessor handles something else).
    amp_norm = getattr(input_preprocessor, 'amp_norm', None)
    if amp_norm is None:
        return 'dataloader'
    return 'model'
