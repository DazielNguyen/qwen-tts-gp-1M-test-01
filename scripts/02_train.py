import os
import sys
import json
import random
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.transformer import MiniAudioLM

# --- 1. QUY HOẠCH TỪ VỰNG (CHUẨN QWEN3) ---
TEXT_VOCAB_SIZE = 151936
AUDIO_VOCAB_SIZE = 4096
SEP_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE       
EOS_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 1   
PAD_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 2   
TOTAL_VOCAB_SIZE = PAD_TOKEN + 1                     

# --- TẤM KHIÊN CHỐNG TRÀN CUDA ---
MAX_SEQ_LENGTH = 1024  

# --- 2. DATASET VÀ BUCKETING ---
class ARDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = torch.tensor(item["text_tokens"], dtype=torch.long)
        audio_cb0 = torch.tensor(item["audio_codes"][0], dtype=torch.long)
        return text, audio_cb0

class BucketBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, drop_last=True):
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        self.lengths = [len(item["text_tokens"]) + len(item["audio_codes"][0]) for item in dataset.data]
        sorted_indices = [i for i, _ in sorted(enumerate(self.lengths), key=lambda x: x[1])]
        self.batches = [sorted_indices[i:i + batch_size] for i in range(0, len(sorted_indices), batch_size)]
        
        if drop_last and len(self.batches[-1]) < batch_size:
            self.batches.pop()

    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)

def ar_collate_fn(batch):
    inputs, targets = [], []
    sep_tensor = torch.tensor([SEP_TOKEN], dtype=torch.long)
    eos_tensor = torch.tensor([EOS_TOKEN], dtype=torch.long)
    
    for text, audio in batch:
        shifted_audio = audio + TEXT_VOCAB_SIZE
        seq_input = torch.cat([text, sep_tensor, shifted_audio])
        
        ignore_text = torch.full_like(text, fill_value=-100)
        seq_target = torch.cat([ignore_text, shifted_audio, eos_tensor])
        
        # 📌 CHẶN LỖI SIGABRT: Cắt gọt các chuỗi dài đột biến vượt ngưỡng 1024
        if seq_input.size(0) > MAX_SEQ_LENGTH:
            seq_input = seq_input[:MAX_SEQ_LENGTH]
            seq_target = seq_target[:MAX_SEQ_LENGTH]
            
        inputs.append(seq_input)
        targets.append(seq_target)
        
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
    
    return inputs_padded, targets_padded

# --- 3. ĐIỀU KHIỂN TRUNG TÂM ---
def main():
    config = {
        'learning_rate': 5e-4,       
        'weight_decay': 0.01,        
        'epochs': 50,                
        'grad_accum_steps': 2,       
        'save_interval': 5,          
        'checkpoint_dir': 'checkpoints/qwen_ar_12hz',
        'batch_size': 16,            
        'num_workers': 0             # 📌 GIẢM TỪ 8 XUỐNG 2 ĐỂ TRÁNH TRÀN SHARED MEMORY BÁO LỖI HỆ ĐIỀU HÀNH
    }
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16", gradient_accumulation_steps=config['grad_accum_steps'])
    
    writer = None
    global_step = 0
    if accelerator.is_local_main_process:
        log_dir = os.path.join(config['checkpoint_dir'], 'logs_ar')
        writer = SummaryWriter(log_dir=log_dir)
        
    accelerator.print("Khởi động Huấn luyện Giai đoạn 1: Auto-Regressive (AR) - Chuẩn 12Hz...")
    
    dataset = ARDataset(jsonl_path="data/processed_dataset_12hz.jsonl")
    accelerator.print(f"Đã nạp {len(dataset)} mẫu dữ liệu siêu tốc.")
    
    batch_sampler = BucketBatchSampler(dataset, batch_size=config['batch_size'], drop_last=True)
    
    dataloader = DataLoader(
        dataset, 
        batch_sampler=batch_sampler, 
        collate_fn=ar_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    model = MiniAudioLM(
        vocab_size=TOTAL_VOCAB_SIZE, 
        d_model=512, nhead=8, num_layers=8, dim_feedforward=2048,
        max_seq_len=MAX_SEQ_LENGTH
    )
    
    if accelerator.is_local_main_process:
        params = sum(p.numel() for p in model.parameters())
        print(f"Mạng Transformer AR: {params:,} tham số (~{params/1e6:.1f}M)")
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    
    total_steps = len(dataloader) * config['epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=int(0.05 * total_steps), num_training_steps=total_steps)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)

    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)
    
    accelerator.print(f"\n🚀 Bắt đầu huấn luyện AR trên {accelerator.num_processes} GPUs!")
    model.train()
    
    for epoch in range(config['epochs']):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['epochs']}", disable=not accelerator.is_local_main_process)
        epoch_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                
                logits = model(inputs) 
                loss = criterion(logits.view(-1, TOTAL_VOCAB_SIZE), targets.view(-1))
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
            epoch_loss += loss.item()
            current_lr = scheduler.get_last_lr()[0]
            progress_bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{current_lr:.2e}")
            
            if accelerator.is_local_main_process:
                writer.add_scalar('Training/Step_Loss', loss.item(), global_step)
                writer.add_scalar('Training/Learning_Rate', current_lr, global_step)
                global_step += 1
            
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            avg_loss = epoch_loss / len(dataloader)
            print(f"Hoàn tất Epoch {epoch+1} | Avg Loss: {avg_loss:.4f}")
            writer.add_scalar('Training/Epoch_Avg_Loss', avg_loss, epoch + 1)
            
            if (epoch + 1) % config['save_interval'] == 0:
                unwrapped_model = accelerator.unwrap_model(model)
                save_path = os.path.join(config['checkpoint_dir'], f"ar_epoch_{epoch+1}.pt")
                torch.save(unwrapped_model.state_dict(), save_path)
                print(f"--> Đã lưu mô hình AR tại: {save_path}")
                
    if accelerator.is_local_main_process and writer is not None:
        writer.close()

if __name__ == "__main__":
    main()