import time

from tsl.predictors import Predictor


class ProfilingPredictor(Predictor):
    step_start = None

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        self.step_start = time.time()
        return batch

    def on_after_backward(self) -> None:
        if self.step_start is not None:
            elapsed = time.time() - self.step_start
            self.loggers[0].log_metric('backward_time', elapsed)
            self.log('backward_time', elapsed, on_step=False,
                     on_epoch=True, batch_size=1)

    def on_train_batch_end(self, outputs, batch, batch_idx, unused=0) -> None:
        if self.step_start is not None:
            elapsed = time.time() - self.step_start
            self.loggers[0].log_metric('train_step_time', elapsed)
            self.log('train_step_time', elapsed, on_step=False,
                     on_epoch=True, batch_size=1)
            self.step_start = None

    def on_validation_batch_end(self, outputs, batch,
                                batch_idx: int, dataloader_idx: int) -> None:
        if self.step_start is not None:
            elapsed = time.time() - self.step_start
            self.log('val_step_time', elapsed, on_step=True,
                     on_epoch=True, batch_size=1)
            self.step_start = None

    def on_test_batch_end(self, outputs, batch,
                          batch_idx: int, dataloader_idx: int) -> None:
        if self.step_start is not None:
            elapsed = time.time() - self.step_start
            self.log('test_step_time', elapsed, on_step=True,
                     on_epoch=True, batch_size=1)
            self.step_start = None
