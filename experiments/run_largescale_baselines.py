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
from tsl.data import SpatioTemporalDataset, AtTimeStepSplitter
from tsl.data.preprocessing import RobustScaler
from tsl.nn.metrics.metrics import MaskedMAE, MaskedMSE, MaskedMRE
from tsl.nn.models import RNNModel, FCRNNModel
from tsl.nn.models.stgn import DCRNNModel
from tsl.utils import TslExperiment, ArgParser, parser_utils
from tsl.utils.parser_utils import str_to_bool

import lib
from lib.datamodule import SubgraphDataModule
from lib.datasets import CEREn, PvUS
from lib.nn.models import GraphWaveNetModel, GatedGraphNetworkMLPModel
from lib.predictors import SubgraphPredictor


def get_model_class(model_str):
    if model_str == 'rnn':
        model = RNNModel
    elif model_str == 'fc_rnn':
        model = FCRNNModel
    # SOTA Models
    elif model_str == 'dcrnn':
        model = DCRNNModel
    elif model_str == 'gwnet':
        model = GraphWaveNetModel
    elif model_str == 'gatedgn':
        model = GatedGraphNetworkMLPModel
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


def configure_parser():
    # Argument parser
    parser = ArgParser(add_help=False)

    # exp config
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument("--model-name", type=str, default='gatedgn')
    parser.add_argument("--config", type=str,
                        default='largescale/gatedgn_pv.yaml')

    # dataset
    parser.add_argument('--dataset-name', type=str, default='pv')
    parser.add_argument('--adj-knn', type=int, default=None)

    # training
    parser.add_argument('--l2-reg', type=float, default=0.)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batches-epoch', type=int, default=50)
    parser.add_argument('--grad-clip-val', type=float, default=5.)

    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--use-lr-schedule', type=str_to_bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--lr-milestones', type=tuple, default=None)
    parser.add_argument('--lr-gamma', type=float, default=None)

    known_args, _ = parser.parse_known_args()
    model_cls = get_model_class(known_args.model_name)
    parser = model_cls.add_model_specific_args(parser)
    parser = SpatioTemporalDataset.add_argparse_args(parser)
    parser = SubgraphDataModule.add_argparse_args(parser)
    parser = SubgraphPredictor.add_argparse_args(parser)
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

    adj = dataset.get_connectivity(threshold=1e-5,
                                   layout='csr' if args.model_name in ['gwnet',
                                                                       'dcrnn'] else 'edge_index',
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

    dm_conf = parser_utils.filter_args(args, SubgraphDataModule,
                                       return_dict=True)
    dm = SubgraphDataModule(
        dataset=torch_dataset,
        scalers={'data': RobustScaler(axis=(0, 1), quantile_range=(10, 90))},
        splitter=get_splitter(args.dataset_name),
        **dm_conf
    )
    dm.setup()

    ########################################
    # predictor                            #
    ########################################
    additional_model_hparams = dict(n_nodes=torch_dataset.n_nodes,
                                    input_size=torch_dataset.n_channels,
                                    input_window_size=torch_dataset.window,
                                    output_size=torch_dataset.n_channels,
                                    window=torch_dataset.window,
                                    horizon=len(range(0, torch_dataset.horizon,
                                                      torch_dataset.horizon_lag)),
                                    exog_size=torch_dataset.input_map.u.n_channels)

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
    scheduler_kwargs = dict(milestones=args.lr_milestones, gamma=args.lr_gamma)
    predictor = SubgraphPredictor(
        model_class=model_cls,
        model_kwargs=model_kwargs,
        optim_class=torch.optim.Adam,
        optim_kwargs={'lr': args.lr, 'weight_decay': args.l2_reg},
        scale_target=args.scale_target,
        loss_fn=loss_fn,
        metrics=metrics,
        scheduler_class=scheduler_class,
        scheduler_kwargs=scheduler_kwargs
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
