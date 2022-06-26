import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

class YoneModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, encoder_weights, in_channels, out_classes, lr=1e-5, threshold=0.5, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch=arch,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
            activation="sigmoid",
            **kwargs,
        )
        self.in_channels = in_channels
        # self.loss_fn = dice_loss()

        self.threshold = threshold
        self.lr = lr

    def forward(self, img):
        # img has already normalized in dataset module!
        img=(img-img.min())/(img.max()-img.min())
        output = self.model(img)
        return output

    def get_inputs(self):
        return (1, self.in_channels, 256, 256) 
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=3e-4, weight_decay=1e-8)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=250, T_mult=2, eta_min=0, last_epoch=-1
        )
        return [optimizer], [scheduler]

    def shared_step(self, batch, stage):

        img = batch["img"]
        assert img.ndim == 4

        h, w = img.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        gt = batch["mask"]
        assert gt.ndim == 4

        output = self.forward(img)

        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(output, gt)

        pred_mask = (output > self.threshold).type(torch.uint8)
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), gt.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }
        
    def get_fuzzy_metrics(self, pred, gt):
        b, c, w, h = pred.shape
        zero_m = torch.zeros((b, c*w*h)).cuda()

        pred = pred.reshape(b, -1)
        gt = gt.reshape(b, -1)

        tp_ = pred.min(gt)
        tp = torch.sum(tp_, dim=1, keepdim=True)

        fp_ = (pred - gt).max(zero_m)
        fp = torch.sum(fp_, dim=1, keepdim=True)

        tn_ = (1-pred).min(1-gt)
        tn = torch.sum(tn_, dim=1, keepdim=True)

        fn_ = (gt - pred).max(zero_m)
        fn = torch.sum(fn_, dim=1, keepdim=True)
        
        rec = torch.mean(tp / (tp + fn))
        prec = torch.mean(tp / (tp + fp))
        f1 = torch.mean(2 * tp / (2 * tp + fp + fn))

        return rec, prec, f1

    def shared_epoch_end(self, outputs, stage):
        
        #TODO
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        rec = smp.metrics.recall(tp, fp, fn, tn, reduction="micro-imagewise")
        prec = smp.metrics.precision(tp, fp, fn, tn, reduction="micro-imagewise")
        f1 = smp.metrics.f1_score(tp, fp, fn, tn, "micro-imagewise")

        metrics = {
            f"{stage}rec": rec,
            f"{stage}prec": prec,
            f"{stage}f1": f1,
        }

        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def training_epoch_end(self, outputs):
        pass

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")