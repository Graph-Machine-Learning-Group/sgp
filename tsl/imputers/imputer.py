from typing import Type, Mapping, Callable, Optional, Union, Tuple, List

import torch
from torch import Tensor
from torch_geometric.data.storage import recursive_apply
from torchmetrics import Metric

from tsl.predictors import Predictor


class Imputer(Predictor):
    r""":class:`~pytorch_lightning.core.LightningModule` to implement imputers.

    Input data should follow the format [batch, steps, nodes, features].

    Args:
        model_class (type): Class of :obj:`~torch.nn.Module` implementing the
            imputer.
        model_kwargs (mapping): Dictionary of arguments to be forwarded to
            :obj:`model_class` at instantiation.
        optim_class (type): Class of :obj:`~torch.optim.Optimizer` implementing
            the optimizer to be used for training the model.
        optim_kwargs (mapping): Dictionary of arguments to be forwarded to
            :obj:`optim_class` at instantiation.
        loss_fn (callable): Loss function to be used for training the model.
        scale_target (bool): Whether to scale target before evaluating the loss.
            The metrics instead will always be evaluated in the original range.
            (default: :obj:`False`)
        whiten_prob (float or list): Randomly mask out a valid datapoint during
            a training step with probability :obj:`whiten_prob`. If a list is
            passed, :obj:`whiten_prob` is sampled from the list for each batch.
            (default: :obj:`0.05`)
        prediction_loss_weight (float): The weight to assign to predictions
            (if any) in the loss. The loss is computed as

            .. math::

                L = \ell(\bar{y}, y, m) + \lambda \sum_i \ell(\hat{y}_i, y, m)

            where :math:`\ell(\bar{y}, y, m)` is the imputation loss,
            :math:`\ell(\bar{y}_i, y, m)` is the forecasting error of prediction
            :math:`\bar{y}_i`, and :math:`\lambda` is :obj:`prediction_loss_weight`.
            (default: :obj:`1.0`)
        impute_only_missing (bool): Whether to impute only missing values in
            inference all the whole sequence.
            (default: :obj:`True`)
        warm_up_steps (int, tuple): Number of steps to be considered as warm up
            stage at the beginning of the sequence. If a tuple is provided, the
            padding is applied both at the beginning and the end of the sequence.
            (default: :obj:`0`)
        metrics (mapping, optional): Set of metrics to be logged during
            train, val and test steps. The metric's name will be automatically
            prefixed with the loop in which the metric is computed (e.g., metric
            :obj:`mae` will be logged as :obj:`train_mae` when evaluated during
            training).
            (default: :obj:`None`)
        scheduler_class (type): Class of :obj:`~torch.optim.lr_scheduler._LRScheduler`
            implementing the learning rate scheduler to be used during training.
            (default: :obj:`None`)
        scheduler_kwargs (mapping): Dictionary of arguments to be forwarded to
            :obj:`scheduler_class` at instantiation.
            (default: :obj:`None`)
    """

    def __init__(self,
                 model_class: Type,
                 model_kwargs: Mapping,
                 optim_class: Type,
                 optim_kwargs: Mapping,
                 loss_fn: Callable,
                 scale_target: bool = False,
                 whiten_prob: Union[float, List[float]] = 0.05,
                 prediction_loss_weight: float = 1.0,
                 impute_only_missing: bool = True,
                 warm_up_steps: Union[int, Tuple[int, int]] = 0,
                 metrics: Optional[Mapping[str, Metric]] = None,
                 scheduler_class: Optional = None,
                 scheduler_kwargs: Optional[Mapping] = None):
        super(Imputer, self).__init__(model_class=model_class,
                                      model_kwargs=model_kwargs,
                                      optim_class=optim_class,
                                      optim_kwargs=optim_kwargs,
                                      loss_fn=loss_fn,
                                      metrics=metrics,
                                      scheduler_class=scheduler_class,
                                      scheduler_kwargs=scheduler_kwargs)
        self.scale_target = scale_target
        self.prediction_loss_weight = prediction_loss_weight
        self.impute_only_missing = impute_only_missing

        if isinstance(whiten_prob, (list, tuple)):
            self.whiten_prob = torch.Tensor(whiten_prob)
        else:
            self.whiten_prob = whiten_prob

        if isinstance(warm_up_steps, int):
            self.warm_up_steps = (warm_up_steps, 0)
        elif isinstance(warm_up_steps, (list, tuple)):
            self.warm_up_steps = tuple(warm_up_steps)
        assert len(self.warm_up_steps) == 2

    def trim_warm_up(self, *args):
        """Trim all tensors in :obj:`args` removing a number of first and last
        steps equals to :obj:`(self.warm_up_steps[0], self.warm_up_steps[1])`,
        respectively."""
        left, right = self.warm_up_steps
        trim = lambda s: s[:, left:s.size(1) - right]
        args = recursive_apply(args, trim)
        if len(args) == 1:
            return args[0]
        return args

    # Imputation data hooks ###################################################

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Rearrange batch for imputation:
            1. Move :obj:`eval_mask` from :obj:`batch.input` to :obj:`batch`
            2. Move :obj:`mask` from :obj:`batch` to :obj:`batch.input`
        """
        # move eval_mask from batch.input to batch
        batch.eval_mask = batch.input.pop('eval_mask')
        # move mask from batch to batch.input
        batch.input.mask = batch.pop('mask')
        # whiten missing values
        if 'x' in batch.input:
            batch.input.x = batch.input.x * batch.input.mask
        return batch

    def on_train_batch_start(self, batch, batch_idx: int,
                             unused: Optional[int] = 0) -> None:
        r"""For every training batch, randomly mask out value with probability
        :math:`p = \texttt{self.whiten\_prob}`. Then, whiten missing values in
         :obj:`batch.input.x`"""
        super(Imputer, self).on_train_batch_start(batch, batch_idx, unused)
        # randomly mask out value with probability p = whiten_prob
        batch.original_mask = mask = batch.input.mask
        p = self.whiten_prob
        if isinstance(p, Tensor):
            p_size = [mask.size(0)] + [1] * (mask.ndim - 1)
            p = p[torch.randint(len(p), p_size)].to(device=mask.device)
        whiten_mask = torch.rand(mask.size(), device=mask.device) > p
        batch.input.mask = mask & whiten_mask
        # whiten missing values
        if 'x' in batch.input:
            batch.input.x = batch.input.x * batch.input.mask

    def _unpack_batch(self, batch):
        transform = batch.get('transform')
        return batch.input, batch.target, batch.eval_mask, transform

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        y = batch.y
        # Make predictions
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)
        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]
        if self.impute_only_missing:
            y_hat = torch.where(batch.mask.bool(), y, y_hat)
        output = dict(y=batch.y, y_hat=y_hat, mask=batch.eval_mask)
        return output

    def shared_step(self, batch, mask):
        y = y_loss = batch.y
        y_hat = y_hat_loss = self.predict_batch(batch, preprocess=False,
                                                postprocess=not self.scale_target)

        if self.scale_target:
            y_loss = batch.transform['y'].transform(y)
            y_hat = batch.transform['y'].inverse_transform(y_hat)

        y_hat_loss, y_loss, mask = self.trim_warm_up(y_hat_loss, y_loss, mask)

        if isinstance(y_hat_loss, (list, tuple)):
            imputation, predictions = y_hat_loss
            y_hat = y_hat[0]
        else:
            imputation, predictions = y_hat_loss, []

        loss = self.loss_fn(imputation, y_loss, mask)
        for pred in predictions:
            pred_loss = self.loss_fn(pred, y_loss, mask)
            loss += self.prediction_loss_weight * pred_loss

        return y_hat.detach(), y, loss

    def training_step(self, batch, batch_idx):

        y_hat, y, loss = self.shared_step(batch, batch.original_mask)

        # Logging
        self.train_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.train_metrics, batch_size=batch.batch_size)
        self.log_loss('train', loss, batch_size=batch.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):

        y_hat, y, val_loss = self.shared_step(batch, batch.mask)

        # Logging
        self.val_metrics.update(y_hat, y, batch.eval_mask)
        self.log_metrics(self.val_metrics, batch_size=batch.batch_size)
        self.log_loss('val', val_loss, batch_size=batch.batch_size)
        return val_loss

    def test_step(self, batch, batch_idx):
        # Compute outputs and rescale
        y_hat = self.predict_batch(batch, preprocess=False, postprocess=True)

        if isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]

        y, training_mask, eval_mask = batch.y, batch.mask, batch.eval_mask
        test_loss = self.loss_fn(y_hat, y, training_mask)  # reconstruction loss

        # Logging
        self.test_metrics.update(y_hat.detach(), y, eval_mask)
        self.log_metrics(self.test_metrics, batch_size=batch.batch_size)
        self.log_loss('test', test_loss, batch_size=batch.batch_size)
        return test_loss

    @staticmethod
    def add_argparse_args(parser, **kwargs):
        parser = Predictor.add_argparse_args(parser)
        parser.add_argument('--whiten-prob', type=float, default=0.05)
        parser.add_argument('--prediction-loss-weight', type=float, default=1.0)
        parser.add_argument('--impute-only-missing', type=bool, default=True)
        parser.add_argument('--warm-up-steps', type=tuple, default=(0, 0))
        return parser
