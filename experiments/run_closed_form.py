import copy
import datetime
import os
import pathlib

import numpy as np
import pytorch_lightning as pl
import tsl
import yaml
from sklearn.linear_model import Ridge
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule, \
    AtTimeStepSplitter
from tsl.data.preprocessing import StandardScaler
from tsl.datasets import MetrLA, PemsBay
from tsl.utils import TslExperiment, ArgParser, parser_utils, numpy_metrics
from tsl.utils.parser_utils import str_to_bool

import lib
from lib.nn.encoders import (SGPSpatialEncoder, GESNEncoder, SGPEncoder)
from lib.utils import encode_dataset


def get_dataset(dataset_name):
    if dataset_name == 'la':
        dataset = MetrLA()
    elif dataset_name == 'bay':
        dataset = PemsBay(mask_zeros=True)
    else:
        raise ValueError(f"Dataset {dataset_name} not available.")
    return dataset


def get_splitter(dataset_name):
    if dataset_name == 'la':
        return AtTimeStepSplitter(first_val_ts=(2012, 5, 25, 16, 00),
                                  last_val_ts=(2012, 6, 4, 3, 20),
                                  first_test_ts=(2012, 6, 4, 4, 20))
    elif dataset_name == 'bay':
        return AtTimeStepSplitter(first_val_ts=(2017, 5, 11, 7, 20),
                                  last_val_ts=(2017, 5, 25, 17, 40),
                                  first_test_ts=(2017, 5, 25, 18, 40))


def get_encoder_class(encoder_name):
    if encoder_name == 'sgp':
        encoder = SGPEncoder
    elif encoder_name == 'space':
        encoder = SGPSpatialEncoder
    elif encoder_name == 'gesn':
        encoder = GESNEncoder
    else:
        raise ValueError(f"Encoder {encoder_name} not available.")
    return encoder


def configure_parser():
    # Argument parser
    parser = ArgParser(add_help=False)

    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--num-threads', type=int, default=1)
    parser.add_argument("--encoder-name", type=str, default='gesn')
    parser.add_argument("--dataset-name", type=str, default='la')
    parser.add_argument("--config", type=str, default=None)

    # training
    parser.add_argument('--preprocess-exogenous', type=str_to_bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--max-nodes-training', type=int, default=None)
    parser.add_argument('--val-len', type=float, default=0.1)
    parser.add_argument('--test-len', type=float, default=0.2)

    known_args, _ = parser.parse_known_args()
    encoder_cls = get_encoder_class(known_args.encoder_name)
    parser = encoder_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataset.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    return parser


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    pl.seed_everything(args.seed)

    tsl.logger.info(f'SEED: {args.seed}')

    encoder_cls = get_encoder_class(args.encoder_name)
    dataset = get_dataset(args.dataset_name)

    tsl.logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(lib.config['logs_dir'],
                          args.dataset_name,
                          exp_name)

    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'exp_config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4,
                  sort_keys=True)

    ########################################
    # data module                          #
    ########################################

    exog_vars = dataset.datetime_encoded('day').values
    exog_vars = {'global_u': exog_vars}

    adj = dataset.get_connectivity(threshold=0.1, layout='edge_index',
                                   include_self=args.encoder_name == 'gesn')

    torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                          connectivity=adj,
                                          mask=dataset.mask,
                                          horizon=args.horizon,
                                          window=args.window,
                                          stride=args.stride,
                                          exogenous=exog_vars)

    dm_conf = parser_utils.filter_args(args, SpatioTemporalDataModule,
                                       return_dict=True)

    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers={'data': StandardScaler(axis=(0, 1))},
        splitter=get_splitter(args.dataset_name),
        pin_memory=False,
        **dm_conf
    )
    dm.setup()

    ########################################
    # dataset encoding                     #
    ########################################

    encoder_input_size = torch_dataset.n_channels
    if args.preprocess_exogenous:
        encoder_input_size += torch_dataset.input_map.u.n_channels
    additional_encoder_hparams = dict(input_size=encoder_input_size)

    encoder_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_encoder_hparams},
        target_cls=encoder_cls,
        return_dict=True)

    tsl.logger.info("Preprocessing dataset...")
    torch_dataset = encode_dataset(
        torch_dataset,
        encoder_class=encoder_cls,
        encoder_kwargs=encoder_kwargs,
        encode_exogenous=args.preprocess_exogenous
    )

    metrics = {
        'mae': numpy_metrics.masked_mae,
        'mse': numpy_metrics.masked_mse,
        'mape': numpy_metrics.masked_mape
    }

    ########################################
    # training                             #
    ########################################

    train_slice_w = dm.train_slice[:-args.horizon]
    x_train, _ = torch_dataset.get_tensors(['data', 'encoded_x'],
                                           preprocess=True, cat_dim=-1,
                                           step_index=train_slice_w)
    x_train = x_train.numpy().reshape(-1, x_train.size(-1))

    val_slice_w = dm.val_slice[:-args.horizon]
    x_val, _ = torch_dataset.get_tensors(['data', 'encoded_x'],
                                         preprocess=True, cat_dim=-1,
                                         step_index=val_slice_w)
    x_val = x_val.numpy().reshape(-1, x_val.size(-1))

    test_slice_w = dm.test_slice[:-args.horizon]
    x_test, _ = torch_dataset.get_tensors(['data', 'encoded_x'],
                                          preprocess=True, cat_dim=-1,
                                          step_index=test_slice_w)
    x_test = x_test.numpy().reshape(-1, x_test.size(-1))

    scaler = torch_dataset.scalers['data'].numpy(inplace=False)

    y_hat_val, y_true_val, mask_v = [], [], []
    y_hat_test, y_true_test, mask_t = [], [], []
    for lag in range(1, args.horizon + 1):
        y_train, _ = torch_dataset.get_tensors(['data'], preprocess=True,
                                               step_index=train_slice_w + lag)
        y_train = y_train.numpy().reshape(-1, torch_dataset.n_channels)
        model = Ridge(alpha=args.l2_reg)
        model.fit(x_train, y_train)

        # val
        y_val, _ = torch_dataset.get_tensors(['data'], preprocess=False,
                                             step_index=val_slice_w + lag)
        y_val = y_val.numpy()
        mask_val = dataset.mask[val_slice_w + lag]

        y_pred_val = model.predict(x_val).reshape(y_val.shape)
        y_pred_val = scaler.inverse_transform(y_pred_val)

        y_test, _ = torch_dataset.get_tensors(['data'], preprocess=False,
                                              step_index=test_slice_w + lag)
        y_test = y_test.numpy()
        mask_test = dataset.mask[test_slice_w + lag]

        y_pred_test = model.predict(x_test).reshape(y_test.shape)
        y_pred_test = scaler.inverse_transform(y_pred_test)

        y_hat_val.append(y_pred_val)
        y_true_val.append(y_val)
        mask_v.append(mask_val)

        y_hat_test.append(y_pred_test)
        y_true_test.append(y_test)
        mask_t.append(mask_test)

        for metric_name, metric_fn in metrics.items():
            err_lag = metric_fn(y_pred_val, y_val, mask_val).item()
            tsl.logger.info(f'val_{metric_name}_at_{lag * 5}: {err_lag:.4f}')

            err_lag = metric_fn(y_pred_test, y_test, mask_test).item()
            tsl.logger.info(f'test_{metric_name}_at_{lag * 5}: {err_lag:.4f}')

    ########################################
    # testing                              #
    ########################################

    y_hat_test = np.stack(y_hat_test, axis=1)
    y_true_test = np.stack(y_true_test, axis=1)
    mask_t = np.stack(mask_t, axis=1)

    y_hat_val = np.stack(y_hat_val, axis=1)
    y_true_val = np.stack(y_true_val, axis=1)
    mask_v = np.stack(mask_v, axis=1)

    for metric_name, metric_fn in metrics.items():
        error = metric_fn(y_hat_val, y_true_val, mask_v).item()
        tsl.logger.info(f'val_{metric_name}: {error:.4f}')

        error = metric_fn(y_hat_test, y_true_test, mask_t).item()
        tsl.logger.info(f'test_{metric_name}: {error:.4f}')


if __name__ == '__main__':
    experiment_parser = configure_parser()
    exp = TslExperiment(run_fn=run_experiment, parser=experiment_parser,
                        config_path=lib.config['config_dir'])
    exp.run()
