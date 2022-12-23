from time import time

import torch
from torch import Tensor
from torch.nn import functional as F
from tsl import logger
from tsl.utils.python_utils import ensure_list


def encode_dataset(
        dataset,
        encoder_class,
        encoder_kwargs,
        encode_exogenous=True,
        keep_raw=False,
        save_path=None
):
    # if preprocess_exogenous is True, preprocess all exogenous
    if isinstance(encode_exogenous, bool):
        preprocess_exogenous = dataset.exogenous.keys() \
            if encode_exogenous else []
    preprocess_exogenous = ensure_list(preprocess_exogenous)

    x, _ = dataset.get_tensors(['data'] + preprocess_exogenous,
                               preprocess=True, cat_dim=-1)

    encoder = encoder_class(**encoder_kwargs)

    start = time()
    encoded_x = encoder(x, edge_index=dataset.edge_index,
                        edge_weight=dataset.edge_weight)
    elapsed = int(time() - start)

    if save_path is not None:
        torch.save(encoded_x, save_path)

    logger.info(
        f"Dataset encoded in {elapsed // 60}:{elapsed % 60:02d} minutes.")

    dataset.add_exogenous('encoded_x', encoded_x, add_to_input_map=False)

    input_map = {'x': ['encoded_x']}
    u = ([] if encode_exogenous else ['u']) + (['data'] if keep_raw else [])
    if len(u):
        input_map['u'] = u
    dataset.set_input_map(input_map)
    return dataset


def self_normalizing_activation(x: Tensor, r: float = 1.0):
    return r * F.normalize(x, p=2, dim=-1)
