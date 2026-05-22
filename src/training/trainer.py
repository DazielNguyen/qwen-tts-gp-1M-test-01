import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


class Trainer:
    def __init__(self, model, dataloader, config, device):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.config = config
        self.device = device

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"],
            betas=(0.9, 0.95),
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config["epochs"])

        self.use_amp = self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self.accum_steps = config.get("grad_accum_steps", 1)
        self.save_interval = config.get("save_interval", 5)
        self.epochs = config["epochs"]

        os.makedirs(config["checkpoint_dir"], exist_ok=True)

    def train(self):
        print(f"\nBat dau huan luyen tren thiet bi: {self.device.type.upper()}")
        print(f"Gradient Accumulation: {self.accum_steps} steps")

        for epoch in range(self.epochs):
            print(f"\n--- Epoch {epoch + 1}/{self.epochs} ---")

            self.model.train()
            self.optimizer.zero_grad()

            progress_bar = tqdm(self.dataloader, desc="Training", leave=False)
            epoch_loss = 0.0

            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                with torch.autocast(
                    device_type=self.device.type,
                    dtype=torch.float16,
                    enabled=self.use_amp,
                ):
                    logits = self.model(inputs)
                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)

                    loss = self.criterion(logits, targets)
                    loss = loss / self.accum_steps

                if self.use_amp:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                if ((batch_idx + 1) % self.accum_steps == 0) or (
                    (batch_idx + 1) == len(self.dataloader)
                ):
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), max_norm=1.0
                        )
                        self.optimizer.step()

                    self.optimizer.zero_grad()
                    self.scheduler.step()

                actual_loss = loss.item() * self.accum_steps
                epoch_loss += actual_loss
                progress_bar.set_postfix(
                    loss=f"{actual_loss:.4f}",
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}",
                )

            avg_loss = epoch_loss / len(self.dataloader)
            print(f"Hoan tat Epoch {epoch + 1} | Avg Loss: {avg_loss:.4f}")

            if (epoch + 1) % self.save_interval == 0:
                self.save_checkpoint(epoch + 1)

    def save_checkpoint(self, epoch):
        path = os.path.join(self.config["checkpoint_dir"], f"model_epoch_{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "scaler_state_dict": self.scaler.state_dict() if self.use_amp else None,
            },
            path,
        )
        print(f"--> Da luu Checkpoint: {path}")
