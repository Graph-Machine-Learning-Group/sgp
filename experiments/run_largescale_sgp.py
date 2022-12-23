import copy
import datetime
import os
import pathlib

import numpy as np
import pytorch_lightning as pl
import torch
import tsl
import yaml
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim.lr_scheduler import MultiStepLR
from tsl.data import SpatioTemporalDataset, SpatioTemporalDataModule, \
    AtTimeStepSplitter
from tsl.data.preprocessing import RobustScaler
from tsl.nn.metrics import MaskedMAE, MaskedMSE, MaskedMRE
from tsl.predictors import Predictor
from tsl.utils import TslExperiment, ArgParser, parser_utils
from tsl.utils.parser_utils import str_to_bool

import lib
from lib.datamodule import SGPDataModule
from lib.datasets import CEREn, PvUS
from lib.nn.encoders import (SGPSpatialEncoder, GESNEncoder, SGPEncoder,
                             SGPTemporalEncoder)
from lib.nn.models import (ESNModel, SGPModel, OnlineSGPModel)
from lib.utils import encode_dataset


def get_model_class(model_str):
    if model_str == 'esn':
        model = ESNModel
    elif model_str == 'sgp':
        model = SGPModel
    elif model_str == 'online_sgp':
        model = OnlineSGPModel
    else:
        raise ValueError(f"Model {model_str} not available.")
    return model


def get_dataset(dataset_name):
    if dataset_name == 'pv':
        dataset = PvUS(freq='30T', mask_zeros=False)
        dataset.reduce_(dataset.index < datetime.datetime(2006, 7, 1))
    elif dataset_name == 'cer':
        dataset = CEREn()
        dataset.reduce_((dataset.index >= datetime.datetime(2010, 1, 1)) &
                        (dataset.index < datetime.datetime(2010, 7, 1)))
    else:
        raise ValueError(f"Dataset {dataset_name} not available.")
    return dataset


def get_splitter(dataset_name):
    if dataset_name == 'pv':
        return AtTimeStepSplitter(first_val_ts=(2006, 5, 1),
                                  last_val_ts=(2006, 5, 31, 6),
                                  first_test_ts=(2006, 6, 1))
    elif dataset_name == 'cer':
        return AtTimeStepSplitter(first_val_ts=(2010, 5, 1),
                                  last_val_ts=(2010, 5, 31, 6),
                                  first_test_ts=(2010, 6, 1))


def get_encoder_class(encoder_name):
    if encoder_name == 'sgp':
        encoder = SGPEncoder
    elif encoder_name == 'time':
        encoder = SGPTemporalEncoder
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
    parser.add_argument("--encoder-name", type=str, default='sgp')
    parser.add_argument("--model-name", type=str, default='sgp')
    parser.add_argument("--config", type=str, default='largescale/sgp_pv.yaml')

    # preprocessing options
    parser.add_argument('--preprocess-exogenous', type=str_to_bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--keep-raw', type=str_to_bool, nargs='?', const=True,
                        default=True)

    # dataset
    parser.add_argument('--dataset-name', type=str, default='cer')
    parser.add_argument('--adj-knn', type=int, default=None)

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?',
                        const=True, default=False)
    parser.opt_list('--lr-milestones', type=tuple, default=(25, 50, 100),
                    options=[(25, 50, 100), (50, 75, 100), (40, 80, 120)],
                    tunable=True)
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=None)
    parser.add_argument('--batches-epoch', type=int, default=-1)
    parser.add_argument('--batch-inference', type=int, default=None)
    parser.add_argument('--grad-clip-val', type=float, default=5.)
    parser.add_argument('--iid-sampling', type=str_to_bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--sgp-preprocessing', type=str_to_bool, nargs='?',
                        const=True, default=False)

    known_args, _ = parser.parse_known_args()
    encoder_cls = get_encoder_class(known_args.encoder_name)
    model_cls = get_model_class(known_args.model_name)
    parser = encoder_cls.add_model_specific_args(parser)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataset.add_argparse_args(parser)
    parser = SpatioTemporalDataModule.add_argparse_args(parser)
    parser = Predictor.add_argparse_args(parser)
    return parser


def run_experiment(args):
    # Set configuration and seed
    args = copy.deepcopy(args)
    if args.seed < 0:
        args.seed = np.random.randint(1e9)
    torch.set_num_threads(max(args.workers, 1))
    pl.seed_everything(args.seed)

    tsl.logger.info(f'SEED: {args.seed}')

    model_cls = get_model_class(args.model_name)
    encoder_cls = get_encoder_class(args.encoder_name)
    dataset = get_dataset(args.dataset_name)

    tsl.logger.info(args)

    ########################################
    # create logdir and save configuration #
    ########################################

    exp_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{args.seed}"
    logdir = os.path.join(lib.config['logs_dir'],
                          args.dataset_name,
                          args.model_name,
                          exp_name)

    # save config for logging
    pathlib.Path(logdir).mkdir(parents=True)
    with open(os.path.join(logdir, 'exp_config.yaml'), 'w') as fp:
        yaml.dump(parser_utils.config_dict_from_args(args), fp, indent=4,
                  sort_keys=True)

    ########################################
    # data module                          #
    ########################################
    # encode time of the day and use it as exogenous variable.
    exog_vars = dataset.datetime_encoded('day').values
    exog_vars = {'global_u': exog_vars}

    adj = dataset.get_connectivity(threshold=1e-5, layout='edge_index',
                                   knn=args.adj_knn, gamma=0.005,
                                   theta=150 if args.adj_knn is None else 50,
                                   binary_weights=False, include_self=False)

    torch_dataset = SpatioTemporalDataset(*dataset.numpy(return_idx=True),
                                          connectivity=adj,
                                          mask=dataset.mask,
                                          horizon=args.horizon,
                                          horizon_lag=args.horizon_lag,
                                          window=args.window,
                                          stride=args.stride,
                                          exogenous=exog_vars)

    dm_conf = parser_utils.filter_args(args, SpatioTemporalDataModule,
                                       return_dict=True)
    dm = SpatioTemporalDataModule(
        dataset=torch_dataset,
        scalers={'data': RobustScaler(axis=(0, 1), quantile_range=(10., 90.))},
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

    # estimate required memory

    req_bytes = 4 * torch_dataset.n_nodes * len(
        torch_dataset) * args.reservoir_layers * args.reservoir_size * (
                        args.receptive_field + 1)
    req_gb = req_bytes / 1024 / 1024 / 1024
    tsl.logger.info(f'Estimated memory size: {req_gb} (GB)')

    torch_dataset = encode_dataset(
        torch_dataset,
        encoder_class=encoder_cls,
        encoder_kwargs=encoder_kwargs,
        encode_exogenous=args.preprocess_exogenous,
        keep_raw=args.keep_raw,
    )

    if args.sgp_preprocessing or args.iid_sampling:
        dm = SGPDataModule(dm, iid_sampling=args.iid_sampling,
                           sgp_preprocessing=args.sgp_preprocessing,
                           receptive_field=args.receptive_field,
                           undirected=args.undirected,
                           bidirectional=args.bidirectional,
                           add_self_loops=args.add_self_loops,
                           global_attr=args.global_attr,
                           batch_inference=args.batch_inference)
        dm.setup()

    ########################################
    # predictor                            #
    ########################################
    order = 1
    if 'receptive_field' in vars(args):
        order += (1 if not args.bidirectional else 2) * args.receptive_field
    if 'global_attr' in vars(args) and args.global_attr:
        order += 1
    if 'reservoir_layers' in vars(args):
        order *= args.reservoir_layers

    itm = next(iter(dm.train_dataloader(batch_size=1)))
    x_size, u_size = itm.x.size(-1), itm.u.size(-1) if 'u' in itm else 0
    additional_model_hparams = dict(n_nodes=torch_dataset.n_nodes,
                                    input_size=x_size,
                                    exog_size=u_size,
                                    output_size=torch_dataset.n_channels,
                                    horizon=len(range(0, torch_dataset.horizon,
                                                      torch_dataset.horizon_lag)),
                                    order=order)

    model_kwargs = parser_utils.filter_args(
        args={**vars(args), **additional_model_hparams},
        target_cls=model_cls,
        return_dict=True)

    loss_fn = MaskedMAE(compute_on_step=True)

    metrics = {'mae': MaskedMAE(compute_on_step=False),
               'mse': MaskedMSE(compute_on_step=False),
               'mre': MaskedMRE(compute_on_step=False),
               'mae_step_1': MaskedMAE(compute_on_step=False, at=0),
               'mae_step_2': MaskedMAE(compute_on_step=False, at=1),
               'mae_step_3': MaskedMAE(compute_on_step=False, at=2),
               'mae_step_4': MaskedMAE(compute_on_step=False, at=3)}

    # setup predictor
    scheduler_class = MultiStepLR if args.use_lr_schedule else None
    scheduler_kwargs = dict(milestones=args.lr_milestones, gamma=0.25)

    predictor = Predictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr, 'weight_decay': args.l2_reg},
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs,
        scale_target=args.scale_target
    )

    ########################################
    # training                             #
    ########################################

    checkpoint_callback = ModelCheckpoint(
        dirpath=logdir,
        save_top_k=1,
        monitor='train_mae',
        mode='min',
    )

    trainer = pl.Trainer(max_epochs=args.epochs,
                         limit_train_batches=args.batches_epoch if args.batches_epoch > 0 else 1.0,
                         default_root_dir=logdir,
                         gpus=1 if torch.cuda.is_available() else None,
                         num_sanity_val_steps=0,
                         gradient_clip_val=args.grad_clip_val,
                         callbacks=[checkpoint_callback])

    trainer.fit(predictor, train_dataloaders=dm.train_dataloader())

    ########################################
    # testing                              #
    ########################################

    predictor.load_model(checkpoint_callback.best_model_path)
    predictor.freeze()

    trainer.test(predictor, dataloaders=dm.test_dataloader())


if __name__ == '__main__':
    experiment_parser = configure_parser()
    exp = TslExperiment(run_fn=run_experiment, parser=experiment_parser,
                        config_path=lib.config['config_dir'])
    exp.run()
