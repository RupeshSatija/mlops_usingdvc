import hydra
import lightning as L
import timm
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from torchmetrics import Accuracy


class DogBreedGenericClassifier(L.LightningModule):
    def __init__(
        self,
        model_name: str,
        num_classes: int,
        pretrained: bool,
        optimizer: dict,
        scheduler: dict,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load configuration
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.optimizer_config = optimizer
        self.scheduler_config = scheduler

        # Load the base model from timm
        self.model = timm.create_model(
            self.model_name, pretrained=self.pretrained, num_classes=self.num_classes
        )

        # Metrics
        self.train_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=self.num_classes)
        self.test_acc = Accuracy(task="multiclass", num_classes=self.num_classes)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log(
            "train/acc", self.train_acc, on_step=True, on_epoch=True, prog_bar=True
        )
        return loss

    def predict_step(self, batch, batch_idx):
        x, _ = batch
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        return preds, probs

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, y)
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, y)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        # optimizer: Optimizer = self.optimizer_config["_target_"](
        #     params=self.parameters(),
        #     **{k: v for k, v in self.optimizer_config.items() if k != "_target_"},
        # )

        # scheduler: _LRScheduler = self.scheduler_config["_target_"](
        #     optimizer=optimizer,
        #     **{k: v for k, v in self.scheduler_config.items() if k != "_target_"},
        # )

        print("~~~~~", self.hparams)
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams["optimizer"]["lr"],
            weight_decay=self.hparams["optimizer"]["weight_decay"],
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            factor=self.hparams["scheduler"]["factor"],
            patience=self.hparams["scheduler"]["patience"],
            min_lr=self.hparams["scheduler"]["min_lr"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
            },
        }

    # ... rest of your class implementation ...


@hydra.main(
    version_base="1.3",
    config_path="../../configs",
    config_name="model/dogbreed_classifier",
)
def main(cfg: DictConfig):
    # This function can be used for testing the model configuration
    model = DogBreedGenericClassifier(cfg)
    print(model)


if __name__ == "__main__":
    main()
