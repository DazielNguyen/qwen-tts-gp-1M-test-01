import os
import sys
import torch
import torchaudio
from transformers import AutoTokenizer

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.transformer import MiniAudioLM

# --- KẾT NỐI VỚI MÃ NGUỒN QWEN LOCAL ---
QWEN_REPO_PATH = os.path.join(os.getcwd(), "Qwen3-TTS")
sys.path.append(QWEN_REPO_PATH)
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

# --- 1. HẰNG SỐ TỪ VỰNG (CHUẨN QWEN3) ---
TEXT_VOCAB_SIZE = 151936
AUDIO_VOCAB_SIZE = 4096
SEP_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE       
EOS_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 1   
PAD_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 2   
TOTAL_VOCAB_SIZE = PAD_TOKEN + 1                     

def load_model(checkpoint_path, device):
    print("Đang tải mô hình Transformer AR (12Hz)...")
    model = MiniAudioLM(
        vocab_size=TOTAL_VOCAB_SIZE, 
        d_model=512, 
        nhead=8, 
        num_layers=8, 
        dim_feedforward=2048,
        max_seq_len=1024 # Đã đồng bộ với lúc Train
    )
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    # Vì file .pt lưu trực tiếp state_dict bằng unwrap_model, ta load thẳng
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def generate_audio_tokens(text, model, tokenizer, device, max_new_tokens=1000):
    # Mã hóa Text bằng Qwen Tokenizer
    text_ids = tokenizer.encode(text)
    text_ids = torch.tensor(text_ids, dtype=torch.long, device=device)
    
    sep_tensor = torch.tensor([SEP_TOKEN], dtype=torch.long, device=device)
    generated_sequence = torch.cat([text_ids, sep_tensor])
    
    print(f"Bắt đầu sinh âm thanh cho câu: '{text}'")
    audio_tokens = []
    
    for step in range(max_new_tokens):
        input_tensor = generated_sequence.unsqueeze(0) 
        logits = model(input_tensor)
        
        next_token_logits = logits[0, -1, :]
        next_token = torch.argmax(next_token_logits)
        
        if next_token == EOS_TOKEN:
            print(f"-> Đã sinh xong tại bước {step} (Gặp EOS).")
            break
            
        generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(0)])
        
        audio_token_val = next_token.item() - TEXT_VOCAB_SIZE
        if 0 <= audio_token_val < AUDIO_VOCAB_SIZE:
            audio_tokens.append(audio_token_val)
        else:
            print(f" Cảnh báo: Sinh ra token rác {next_token.item()}, bỏ qua.")

    return torch.tensor(audio_tokens, dtype=torch.long, device=device)

def decode_to_wav(audio_tokens, output_path, device):
    print("Đang tải bộ giải mã Qwen3 Audio Tokenizer (12Hz)...")
    encodec = Qwen3TTSTokenizerV2Model.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz").to(device)
    encodec.eval()
    
    if len(audio_tokens) == 0:
        print("Lỗi: Không có token âm thanh nào được sinh ra!")
        return

    seq_len = len(audio_tokens)
    
    # --- TẠO MA TRẬN 16 CODEBOOKS ---
    # Shape yêu cầu của Qwen: [batch_size, seq_len, num_quantizers]
    audio_codes = torch.zeros((1, seq_len, 16), dtype=torch.long, device=device)
    
    # Nhét token dự đoán được vào Codebook 0. Các Codebook còn lại giữ nguyên số 0
    audio_codes[0, :, 0] = audio_tokens
    
    print("Đang giải mã Audio Token -> Waveform...")
    with torch.no_grad():
        output = encodec.decode(audio_codes, return_dict=True)
        # Lấy waveform của phần tử đầu tiên trong batch
        wav_output = output.audio_values[0] 
        
    wav_tensor = wav_output.cpu().unsqueeze(0) # Đưa về [1, samples] cho torchaudio
    target_sr = encodec.get_output_sample_rate() # Tự động lấy tần số gốc (chuẩn 24000Hz)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, wav_tensor, target_sr)
    print(f"✅ Đã lưu file âm thanh thô tại: {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    text_input = "Hello world, this is a test of the Qwen TTS prototype."
    # 📌 CẬP NHẬT TÊN FILE THEO ĐÚNG THƯ MỤC CỦA LỚP 02
    checkpoint_file = "checkpoints/qwen_ar_12hz/ar_epoch_5.pt" 
    output_wav = "output_samples/test_ar_12hz_epoch_5.wav"
    
    if not os.path.exists(checkpoint_file):
        print(f"❌ Chưa tìm thấy {checkpoint_file}. Hãy đợi model train đến Epoch 5 nhé!")
        return
        
    # Khởi tạo Qwen Text Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True)
    
    model = load_model(checkpoint_file, device)
    audio_tokens = generate_audio_tokens(text_input, model, tokenizer, device)
    decode_to_wav(audio_tokens, output_wav, device)

if __name__ == "__main__":
    main()