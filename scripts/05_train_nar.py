import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Giả định bạn có kiến trúc MiniAudioNAR trong file model
from src.models.transformer import MiniAudioNAR 

# --- 1. CẤU HÌNH ---
AUDIO_VOCAB_SIZE = 4096
NUM_NAR_CODEBOOKS = 15  # Dự đoán từ codebook 1 đến 15
TOTAL_VOCAB_SIZE = AUDIO_VOCAB_SIZE + 1 # +1 cho padding

class NARDataset(Dataset):
    def __init__(self, jsonl_path):
        self.data = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Input: Codebook 0 (ngữ điệu chính)
        audio_codes = torch.tensor(item["audio_codes"], dtype=torch.long)
        cb0 = audio_codes[0] 
        # Target: 15 Codebooks còn lại
        cbs_1_15 = audio_codes[1:16]
        return cb0, cbs_1_15

def nar_collate_fn(batch):
    cb0s, cbs_1_15 = zip(*batch)
    # Pad Codebook 0
    cb0s_padded = nn.utils.rnn.pad_sequence(cb0s, batch_first=True, padding_value=0)
    # Pad 15 Codebooks (shape: [batch, 15, seq_len])
    # Cần permute về [batch, seq_len, 15] để pad
    cbs_1_15_padded = []
    for cb in cbs_1_15:
        cbs_1_15_padded.append(cb.transpose(0, 1))
    
    cbs_padded = nn.utils.rnn.pad_sequence(cbs_1_15_padded, batch_first=True, padding_value=0)
    return cb0s_padded, cbs_padded.transpose(1, 2) # Trả về [batch, 15, seq_len]

def main():
    config = {
        'learning_rate': 2e-4,
        'epochs': 30,
        'batch_size': 16,
        'checkpoint_dir': 'checkpoints/qwen_nar_12hz',
        'num_workers': 0
    }
    
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    accelerator = Accelerator(mixed_precision="fp16")
    
    dataset = NARDataset("data/processed_dataset_12hz.jsonl")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], collate_fn=nar_collate_fn, num_workers=config['num_workers'])
    
    # Mạng NAR: Input là [seq, 1], Output là [seq, 15]
    model = MiniAudioNAR(
        vocab_size=AUDIO_VOCAB_SIZE,
        num_target_codebooks=NUM_NAR_CODEBOOKS,
        d_model=512,
        num_layers=6
    )
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    
    # Loss: CrossEntropy cho mỗi codebook
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    writer = SummaryWriter(os.path.join(config['checkpoint_dir'], 'logs'))
    global_step = 0

    model.train()
    for epoch in range(config['epochs']):
        for cb0, cbs_1_15 in tqdm(dataloader, desc=f"NAR Epoch {epoch+1}"):
            optimizer.zero_grad()
            
            # Predict: [batch, 15, seq_len]
            logits = model(cb0)
            
            # Tính loss trên cả 15 codebooks
            loss = criterion(logits, cbs_1_15)
            
            accelerator.backward(loss)
            optimizer.step()
            
            if accelerator.is_local_main_process:
                writer.add_scalar('NAR/Loss', loss.item(), global_step)
                global_step += 1
        
        # Save model
        if (epoch + 1) % 5 == 0:
            unwrapped = accelerator.unwrap_model(model)
            torch.save(unwrapped.state_dict(), f"{config['checkpoint_dir']}/nar_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    main()