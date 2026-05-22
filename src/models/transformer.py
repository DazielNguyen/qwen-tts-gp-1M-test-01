import torch
import torch.nn as nn
import math

class MiniAudioLM(nn.Module):
    def __init__(self, 
                 vocab_size=51284,      # [CẬP NHẬT] Text(50257) + Audio(1024) + Special(3)
                 d_model=512,           # [CẬP NHẬT] Chuẩn prototype 50M-100M
                 nhead=8,               # [CẬP NHẬT] Số head tăng lên để học đa dạng đặc trưng
                 num_layers=8,          # [CẬP NHẬT] Số layer mạng sâu hơn
                 dim_feedforward=2048,  # [CẬP NHẬT] Thường là d_model * 4
                 max_seq_len=4096):     # [CẬP NHẬT] Tăng độ dài chuỗi vì Text + Audio nối nhau rất dài
        super().__init__()
        self.d_model = d_model
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        # --- NÂNG CẤP KIẾN TRÚC LLM HIỆN ĐẠI ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=0.1, 
            activation="gelu",       # 1. Đổi ReLU thành GELU (Chuẩn Qwen/LLaMA)
            batch_first=True,
            norm_first=True          # 2. Pre-LayerNorm: Cực kỳ quan trọng để chống vỡ Loss
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def generate_causal_mask(self, sz, device):
        # 3. Sử dụng hàm tạo mask tối ưu của PyTorch (Hỗ trợ FlashAttention phần cứng)
        return nn.Transformer.generate_square_subsequent_mask(sz, device=device)

    def forward(self, x):
        seq_len = x.size(1)
        
        # Đảm bảo đầu vào không vượt quá giới hạn position
        if seq_len > self.position_embedding.num_embeddings:
            raise ValueError(f"Sequence length {seq_len} vượt quá max_seq_len {self.position_embedding.num_embeddings}")
            
        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0)
        
        x = self.token_embedding(x) * math.sqrt(self.d_model) + self.position_embedding(positions)
        
        # Tạo Causal Mask 
        causal_mask = self.generate_causal_mask(seq_len, x.device)
        
        # Truyền qua Transformer với is_causal=True để PyTorch tối ưu bộ nhớ
        out = self.transformer(x, mask=causal_mask, is_causal=True)
        return self.lm_head(out)