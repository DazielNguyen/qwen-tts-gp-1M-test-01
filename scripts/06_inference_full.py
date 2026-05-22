import os
import sys
import torch
import torchaudio
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.transformer import MiniAudioLM, MiniAudioNAR
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

# --- HẰNG SỐ ---
TEXT_VOCAB_SIZE = 151936
AUDIO_VOCAB_SIZE = 4096

@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Tải các thành phần
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True)
    
    # AR Model (Dự đoán CB 0)
    ar_model = MiniAudioLM(vocab_size=TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 3, d_model=512, nhead=8, num_layers=8, dim_feedforward=2048, max_seq_len=1024).to(device)
    ar_model.load_state_dict(torch.load("checkpoints/qwen_ar_12hz/ar_epoch_50.pt", map_location=device))
    ar_model.eval()
    
    # NAR Model (Dự đoán CB 1-15)
    nar_model = MiniAudioNAR(vocab_size=AUDIO_VOCAB_SIZE, num_target_codebooks=15, d_model=512, num_layers=6).to(device)
    nar_model.load_state_dict(torch.load("checkpoints/qwen_nar_12hz/nar_epoch_30.pt", map_location=device))
    nar_model.eval()
    
    # Decoder (Qwen EnCodec)
    codec = Qwen3TTSTokenizerV2Model.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz").to(device)
    codec.eval()

    # 2. Sinh Audio (Pipeline)
    text = "Chào bạn, đây là hệ thống Qwen TTS 12 Hertz hoàn chỉnh."
    print(f"Generating for: {text}")
    
    # Step A: AR sinh Codebook 0
    # (Dùng logic generate_audio_tokens từ file 03 đã tối ưu)
    cb0 = generate_cb0(text, ar_model, tokenizer, device)
    
    # Step B: NAR sinh 15 Codebooks còn lại
    cbs_1_15 = nar_model(cb0.unsqueeze(0)) # [1, 15, seq]
    
    # Step C: Ghép lại thành 16 Codebooks
    # cb0: [1, seq], cbs_1_15: [1, 15, seq] -> cat -> [1, 16, seq]
    audio_codes = torch.cat([cb0.unsqueeze(0).unsqueeze(0), cbs_1_15], dim=1).transpose(1, 2)
    
    # Step D: Decode
    output = codec.decode(audio_codes, return_dict=True)
    wav = output.audio_values[0].cpu()
    
    torchaudio.save("output_final.wav", wav.unsqueeze(0), codec.get_output_sample_rate())
    print("✅ Hoàn thành! File nằm tại: output_final.wav")

def generate_cb0(text, model, tokenizer, device):
    # Logic tương tự script 03 của bạn
    text_ids = torch.tensor(tokenizer.encode(text), device=device)
    # ... (Giữ nguyên logic sinh sequence cho đến khi EOS) ...
    return cb0_tokens

if __name__ == "__main__":
    main()