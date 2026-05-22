import sys
import os
import glob
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from accelerate import Accelerator
from torch.optim import AdamW
from copy import deepcopy
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.transformer import MiniAudioLM

# --- 1. HẰNG SỐ TỪ VỰNG ---
TEXT_VOCAB_SIZE = 50257
AUDIO_VOCAB_SIZE = 1024
SEP_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE       
EOS_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 1   
PAD_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 2   
TOTAL_VOCAB_SIZE = PAD_TOKEN + 1                     

# --- 2. DATASET DPO (CHOSEN vs REJECTED) ---
class DPODataset(Dataset):
    def __init__(self, data_dir):
        # Giả định bạn đã chuẩn bị các file .pt chứa dict: {"text": ..., "chosen_audio": ..., "rejected_audio": ...}
        self.files = glob.glob(f"{data_dir}/*.pt")
        print(f"[DPO] Tìm thấy {len(self.files)} file Preference Pairs.")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx], map_location="cpu", weights_only=False)
        return data["text_tokens"], data["chosen_audio"], data["rejected_audio"]

def dpo_collate_fn(batch):
    inputs_chosen, targets_chosen = [], []
    inputs_rejected, targets_rejected = [], []
    sep_tensor = torch.tensor([SEP_TOKEN], dtype=torch.long)
    eos_tensor = torch.tensor([EOS_TOKEN], dtype=torch.long)
    
    for text, chosen, rejected in batch:
        shifted_chosen = chosen + TEXT_VOCAB_SIZE
        shifted_rejected = rejected + TEXT_VOCAB_SIZE
        ignore_text = torch.full_like(text, fill_value=-100)
        
        # Xử lý Chosen (Bản Tốt)
        inputs_chosen.append(torch.cat([text, sep_tensor, shifted_chosen]))
        targets_chosen.append(torch.cat([ignore_text, shifted_chosen, eos_tensor]))
        
        # Xử lý Rejected (Bản Kém)
        inputs_rejected.append(torch.cat([text, sep_tensor, shifted_rejected]))
        targets_rejected.append(torch.cat([ignore_text, shifted_rejected, eos_tensor]))
        
    batch_dict = {
        "chosen_inputs": pad_sequence(inputs_chosen, batch_first=True, padding_value=PAD_TOKEN),
        "chosen_targets": pad_sequence(targets_chosen, batch_first=True, padding_value=-100),
        "rejected_inputs": pad_sequence(inputs_rejected, batch_first=True, padding_value=PAD_TOKEN),
        "rejected_targets": pad_sequence(targets_rejected, batch_first=True, padding_value=-100),
    }
    return batch_dict

# --- 3. HÀM TÍNH XÁC SUẤT CHUỖI (LOG PROBS) ---
def get_batch_logps(logits, labels, ignore_index=-100):
    """
    Tính tổng log probability của các token thực tế được sinh ra.
    Logits shape: [batch, seq_len, vocab_size]
    Labels shape: [batch, seq_len]
    """
    # Trượt đi 1 vị trí vì mô hình dự đoán token tiếp theo
    logits_shifted = logits[:, :-1, :]
    labels_shifted = labels[:, 1:].clone()
    
    # Tính log softmax
    log_probs = F.log_softmax(logits_shifted, dim=-1)
    
    # Rút trích xác suất tại đúng index của labels
    per_token_logps = torch.gather(log_probs, dim=2, index=labels_shifted.unsqueeze(2)).squeeze(2)
    
    # Loại bỏ các vị trí -100 (Text và Padding)
    loss_mask = (labels_shifted != ignore_index)
    
    # Tính tổng log_prob cho toàn bộ chuỗi Audio
    return (per_token_logps * loss_mask).sum(-1)

# --- 4. ĐIỀU KHIỂN TRUNG TÂM ---
def main():
    print("Khởi động Tinh chỉnh Ngữ điệu (DPO) cho AR Model...")
    
    config = {
        'learning_rate': 1e-6,       # DPO cần LR cực nhỏ để không phá hỏng mô hình
        'beta': 0.1,                 # Hệ số phạt DPO (Thường từ 0.1 đến 0.5)
        'epochs': 3,                 # DPO chỉ cần train vài epoch
        'batch_size': 2,             # Batch rất nhỏ vì tốn x2 bộ nhớ (Policy + Ref model)
        'checkpoint_dir': 'checkpoints/qwen_dpo',
        'pretrained_ar_path': 'checkpoints/qwen_proto/ar_epoch_50.pt' # Trọng số từ Giai đoạn 1
    }
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    accelerator = Accelerator(mixed_precision="fp16")
    device = accelerator.device
    
    # 1. Tải Mô hình Chính (Policy Model) - Sẽ được huấn luyện
    model = MiniAudioLM(vocab_size=TOTAL_VOCAB_SIZE, d_model=512, nhead=8, num_layers=8, dim_feedforward=2048, max_seq_len=4096)
    model.load_state_dict(torch.load(config['pretrained_ar_path'], map_location="cpu"))
    
    # 2. Tải Mô hình Tham chiếu (Reference Model) - Bị đóng băng hoàn toàn
    ref_model = deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False
        
    dataset = DPODataset(data_dir="data/dpo_pairs")
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, collate_fn=dpo_collate_fn)
    
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    
    # Đưa vào Accelerator (Chỉ đưa policy model, không đưa ref_model)
    model, optimizer, dataloader = accelerator.prepare(model, optimizer, dataloader)
    ref_model = ref_model.to(device)
    
    # --- VÒNG LẶP DPO ---
    for epoch in range(config['epochs']):
        model.train()
        progress_bar = tqdm(dataloader, desc=f"DPO Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        
        for batch in progress_bar:
            optimizer.zero_grad()
            
            # --- TÍNH LOGITS TỪ POLICY MODEL ---
            pi_chosen_logits = model(batch["chosen_inputs"])
            pi_rejected_logits = model(batch["rejected_inputs"])
            
            pi_chosen_logps = get_batch_logps(pi_chosen_logits, batch["chosen_targets"])
            pi_rejected_logps = get_batch_logps(pi_rejected_logits, batch["rejected_targets"])
            
            # --- TÍNH LOGITS TỪ REF MODEL (KHÔNG CẦN GRADIENT) ---
            with torch.no_grad():
                ref_chosen_logits = ref_model(batch["chosen_inputs"])
                ref_rejected_logits = ref_model(batch["rejected_inputs"])
                
                ref_chosen_logps = get_batch_logps(ref_chosen_logits, batch["chosen_targets"])
                ref_rejected_logps = get_batch_logps(ref_rejected_logits, batch["rejected_targets"])
                
            # --- TÍNH TOÁN DPO LOSS ---
            pi_logratios = pi_chosen_logps - pi_rejected_logps
            ref_logratios = ref_chosen_logps - ref_rejected_logps
            
            logits_diff = pi_logratios - ref_logratios
            
            # Công thức Bradley-Terry Loss
            loss = -F.logsigmoid(config['beta'] * logits_diff).mean()
            
            accelerator.backward(loss)
            optimizer.step()
            
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            
        # Lưu Checkpoint
        accelerator.wait_for_everyone()
        if accelerator.is_local_main_process:
            save_path = os.path.join(config['checkpoint_dir'], f"dpo_epoch_{epoch+1}.pt")
            torch.save(accelerator.unwrap_model(model).state_dict(), save_path)
            print(f"--> Đã lưu mô hình DPO tại: {save_path}")

if __name__ == "__main__":
    main()

