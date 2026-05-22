import os
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from accelerate import Accelerator

class Trainer:
    def __init__(self, model, dataloader, config):
        self.config = config
        
        # 1. Khởi tạo bộ siêu tốc Accelerator
        self.accelerator = Accelerator(
            mixed_precision="fp16", 
            gradient_accumulation_steps=config.get('grad_accum_steps', 1)
        )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['learning_rate'], 
            weight_decay=config['weight_decay'],
            betas=(0.9, 0.95)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config['epochs'])
        
        # 2. Ủy quyền toàn bộ Model, Optimizer, DataLoader cho Accelerator quản lý
        self.model, self.optimizer, self.dataloader, self.scheduler = self.accelerator.prepare(
            model, self.optimizer, dataloader, self.scheduler
        )
        
        self.save_interval = config.get('save_interval', 5)
        self.epochs = config['epochs']
        os.makedirs(config['checkpoint_dir'], exist_ok=True)

    def train(self):
        # Chỉ in ra dòng này ở GPU chính để tránh bị in 2 lần
        self.accelerator.print(f"\n🚀 Bắt đầu huấn luyện trên {self.accelerator.num_processes} GPUs!")
        
        for epoch in range(self.epochs):
            self.model.train()
            
            # Vô hiệu hóa thanh TQDM ở GPU phụ để log không bị rối
            progress_bar = tqdm(self.dataloader, desc=f"Epoch {epoch+1}/{self.epochs}", 
                                disable=not self.accelerator.is_local_main_process)
            epoch_loss = 0.0
            
            for batch_idx, (inputs, targets) in enumerate(progress_bar):
                with self.accelerator.accumulate(self.model):
                    self.optimizer.zero_grad()
                    
                    logits = self.model(inputs)
                    logits = logits.view(-1, logits.size(-1))
                    targets = targets.view(-1)
                    
                    loss = self.criterion(logits, targets)
                    
                    # Backward bằng Accelerator (Tự động scale fp16)
                    self.accelerator.backward(loss)
                    
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        
                    self.optimizer.step()
                    self.scheduler.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix(
                    loss=f"{loss.item():.4f}", 
                    lr=f"{self.scheduler.get_last_lr()[0]:.2e}"
                )
                # Ghi log từng bước vào TensorBoard
                if self.accelerator.is_local_main_process:
                    self.writer.add_scalar('Training/Step_Loss', loss.item(), self.global_step)
                    self.writer.add_scalar('Training/Learning_Rate', self.scheduler.get_last_lr()[0], self.global_step)
                    self.global_step += 1


            # Chờ cả 2 GPU chạy xong epoch mới đi tiếp
            self.accelerator.wait_for_everyone()
            
            if self.accelerator.is_local_main_process:
                avg_loss = epoch_loss / len(self.dataloader)
                print(f"Hoàn tất Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
                
                # Ghi log trung bình của Epoch
                self.writer.add_scalar('Training/Epoch_Avg_Loss', avg_loss, epoch + 1)

                if (epoch + 1) % self.save_interval == 0:
                    self.save_checkpoint(epoch + 1)

        # Đóng writer khi train xong
        if self.accelerator.is_local_main_process:
            self.writer.close()

    def save_checkpoint(self, epoch):
        path = os.path.join(self.config['checkpoint_dir'], f"model_epoch_{epoch}.pt")
        # Bóc lớp vỏ bọc DDP của Accelerator ra trước khi lưu để lúc sau dễ load
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        torch.save({
            'epoch': epoch,
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, path)
        print(f"--> Đã lưu Checkpoint: {path}")