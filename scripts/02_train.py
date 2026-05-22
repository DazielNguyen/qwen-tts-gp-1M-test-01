import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader

# Import Model và Trainer của bạn
from src.models.transformer import MiniAudioLM
from src.training.trainer import Trainer

# --- 1. QUY HOẠCH TỪ VỰNG (VOCABULARY MATH) ---
TEXT_VOCAB_SIZE = 50257
AUDIO_VOCAB_SIZE = 1024
SEP_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE       # 51281
EOS_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 1   # 51282
PAD_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 2   # 51283
TOTAL_VOCAB_SIZE = PAD_TOKEN + 1                     # 51284

# --- 2. DATASET VÀ BATCHING ---
class ARDataset(Dataset):
    def __init__(self, data_dir):
        self.files = glob.glob(f"{data_dir}/*.pt")
        print(f"Tìm thấy {len(self.files)} file dữ liệu.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # Đọc file dữ liệu đã chuẩn bị từ Script 1
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        text = data["text_tokens"]  
        audio_cb0 = data["audio_codes"][0] # AR Mode: Chỉ học Codebook 0
        return text, audio_cb0

def ar_collate_fn(batch):
    inputs, targets = [], []
    sep_tensor = torch.tensor([SEP_TOKEN], dtype=torch.long)
    eos_tensor = torch.tensor([EOS_TOKEN], dtype=torch.long)
    
    for text, audio in batch:
        shifted_audio = audio + TEXT_VOCAB_SIZE
        
        # Input: [Text] -> [SEP] -> [Audio]
        seq_input = torch.cat([text, sep_tensor, shifted_audio])
        
        # Target: [-100] -> [Audio] -> [EOS]
        ignore_text = torch.full_like(text, fill_value=-100)
        seq_target = torch.cat([ignore_text, shifted_audio, eos_tensor])
        
        inputs.append(seq_input)
        targets.append(seq_target)
        
    inputs_padded = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True, padding_value=PAD_TOKEN)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True, padding_value=-100)
    
    return inputs_padded, targets_padded

# --- 3. ĐIỀU KHIỂN TRUNG TÂM ---
def main():
    print("Khởi động Huấn luyện Giai đoạn 1: Auto-Regressive (AR)...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 📌 BẢNG CẤU HÌNH (CONFIGURATION DICT)
    # Truyền cục config này sang trainer.py
    config = {
        'learning_rate': 5e-4,       # Tốc độ học khởi điểm (5e-4 là chuẩn cho Transformer nhỏ)
        'weight_decay': 0.01,        # Lọc nhiễu, chống overfit
        'epochs': 50,                # Số vòng lặp qua dataset
        'grad_accum_steps': 16,      # Tăng từ 4 lên 16     # Tích lũy gradient (Mô phỏng batch_size x 4)
        'save_interval': 5,          # Lưu checkpoint mỗi 5 epoch
        'checkpoint_dir': 'checkpoints/qwen_proto',
        'batch_size': 4,             # Giảm từ 16 xuống 4    # Số mẫu mỗi batch (Tinh chỉnh theo VRAM của V100)
        'num_workers': 8             # Số luồng CPU đọc file
    }
    
        

    # Khởi tạo DataLoader
    dataset = ARDataset(data_dir="data/tokens_16cb")
    dataloader = DataLoader(
        dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=ar_collate_fn,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    # Khởi tạo Model
    model = MiniAudioLM(
        vocab_size=TOTAL_VOCAB_SIZE, 
        d_model=512, 
        nhead=8, 
        num_layers=8, 
        dim_feedforward=2048,
        max_seq_len=4096
    )
    
    print(f"Mạng Transformer AR: {sum(p.numel() for p in model.parameters()):,} tham số (~{sum(p.numel() for p in model.parameters())/1e6:.1f}M)")
    
    # Kích hoạt Trainer
    trainer = Trainer(model=model, dataloader=dataloader, config=config, device=device)
    
    # Bắt đầu chạy
    trainer.train()

if __name__ == "__main__":
    main()