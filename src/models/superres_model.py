from typing import Any, List

import torch
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics import SSIM, PSNR

from src.models.pytorch_models.get_model import get_model
from src.losses.get_loss import get_loss


class SuperResLitModel(LightningModule):
    """
    Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
            self,
            lr: float = 0.001,
            weight_decay: float = 0.0005,
            loss_name: str = 'L1'
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.model = get_model(model_name='UNet', hparams=self.hparams)

        # loss function
        self.criterion = get_loss(loss_name=self.hparams.loss_name)

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_ssim = SSIM()
        self.val_ssim = SSIM()
        self.test_ssim = SSIM()

        self.train_psnr = PSNR()
        self.val_psnr = PSNR()
        self.test_psnr = PSNR()

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def step(self, batch: Any):
        x, y = batch
        output = self.forward(x)
        loss = self.criterion(output, y)
        return loss, output, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        ssim = self.train_ssim(preds.detach().clone(), targets)
        psnr = self.train_psnr(preds.detach().clone(), targets)

        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/SSIM", ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/PSNR", psnr, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        ssim = self.val_ssim(preds.detach().clone(), targets)
        psnr = self.val_psnr(preds.detach().clone(), targets)

        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val/SSIM", ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/PSNR", psnr, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        ssim = self.test_ssim(preds.detach().clone(), targets)
        psnr = self.test_psnr(preds.detach().clone(), targets)

        self.log("test/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("test/SSIM", ssim, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/PSNR", psnr, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch!
        self.train_ssim.reset()
        self.test_ssim.reset()
        self.val_ssim.reset()

        self.train_psnr.reset()
        self.test_psnr.reset()
        self.val_psnr.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        return torch.optim.Adam(
            params=self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay
        )
