import torch
import torchaudio
import os
from transformers import AutoTokenizer, EncodecModel
from src.models.transformer import MiniAudioLM

# --- 1. HẰNG SỐ TỪ VỰNG ---
TEXT_VOCAB_SIZE = 50257
AUDIO_VOCAB_SIZE = 1024
SEP_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE       # 51281
EOS_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 1   # 51282
PAD_TOKEN = TEXT_VOCAB_SIZE + AUDIO_VOCAB_SIZE + 2   # 51283
TOTAL_VOCAB_SIZE = PAD_TOKEN + 1                     # 51284

def load_model(checkpoint_path, device):
    print("Đang tải mô hình Transformer AR...")
    model = MiniAudioLM(
        vocab_size=TOTAL_VOCAB_SIZE, 
        d_model=512, 
        nhead=8, 
        num_layers=8, 
        dim_feedforward=2048,
        max_seq_len=4096
    )
    
    # Load trọng số từ file Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Lấy state_dict, nếu checkpoint được lưu bởi dictionary của Trainer thì lấy 'model_state_dict'
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def generate_audio_tokens(text, model, tokenizer, device, max_new_tokens=1000):
    # Mã hóa Text bằng BPE
    text_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0).to(device)
    
    # Khởi tạo chuỗi đầu vào: [Text] + [SEP]
    sep_tensor = torch.tensor([SEP_TOKEN], dtype=torch.long, device=device)
    generated_sequence = torch.cat([text_ids, sep_tensor])
    
    print(f"Bắt đầu sinh âm thanh cho câu: '{text}'")
    audio_tokens = []
    
    # Vòng lặp Auto-Regressive: Dự đoán từng token một từ trái sang phải
    for step in range(max_new_tokens):
        # Đưa chuỗi hiện tại vào model
        input_tensor = generated_sequence.unsqueeze(0) # [1, seq_len]
        logits = model(input_tensor)
        
        # Lấy token cuối cùng được dự đoán
        next_token_logits = logits[0, -1, :]
        
        # Ở đây dùng Greedy Search (lấy token có xác suất cao nhất)
        next_token = torch.argmax(next_token_logits)
        
        # Nếu model dự đoán xong (chạm EOS) -> Dừng lại
        if next_token == EOS_TOKEN:
            print(f" Đã sinh xong tại bước {step} (Gặp EOS).")
            break
            
        # Thêm token vào chuỗi để dự đoán bước tiếp theo
        generated_sequence = torch.cat([generated_sequence, next_token.unsqueeze(0)])
        
        # Lọc ra giá trị Audio (nhớ trừ đi TEXT_VOCAB_SIZE)
        audio_token_val = next_token.item() - TEXT_VOCAB_SIZE
        # Kiểm tra tính hợp lệ
        if 0 <= audio_token_val < AUDIO_VOCAB_SIZE:
            audio_tokens.append(audio_token_val)
        else:
            print(f" Cảnh báo: Sinh ra token rác {next_token.item()}, bỏ qua.")

    return torch.tensor(audio_tokens, dtype=torch.long, device=device)

def decode_to_wav(audio_tokens, output_path, device):
    print("Đang tải bộ giải mã EnCodec...")
    encodec = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    encodec.eval()
    
    if len(audio_tokens) == 0:
        print("Lỗi: Không có token âm thanh nào được sinh ra!")
        return

    # Hugging Face EnCodec yêu cầu input có shape: [batch, num_quantizers, channels, time_steps]
    # Vì AR chỉ sinh Codebook 0, ta cấu trúc lại thành [1, 1, 1, seq_len]
    encodec_input = audio_tokens.view(1, 1, 1, -1)
    
    print("Đang giải mã Audio Token -> Waveform...")
    with torch.no_grad():
        # Tham số audio_scales có thể truyền None khi sinh từ token thuần
        wav_output = encodec.decode(encodec_input, [None])[0]
        
    # Shape wav_output: [1, 1, samples]
    wav_tensor = wav_output.squeeze(0).cpu()
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torchaudio.save(output_path, wav_tensor, 24000)
    print(f"Đã lưu file âm thanh tại: {output_path}")

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- CẤU HÌNH INFERENCE ---
    text_input = "Hello world, this is a test of the Qwen TTS prototype."
    # Bạn chỉ cần thay tên file checkpoint ở đây khi có checkpoint mới (VD: sau epoch 5)
    checkpoint_file = "checkpoints/qwen_proto/model_epoch_5.pt" 
    output_wav = "output_samples/test_epoch_5.wav"
    
    if not os.path.exists(checkpoint_file):
        print(f"Chưa tìm thấy {checkpoint_file}. Hãy đợi model lưu xong rồi chạy nhé!")
        return
        
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # 1. Tải mô hình
    model = load_model(checkpoint_file, device)
    
    # 2. Sinh Audio Tokens
    audio_tokens = generate_audio_tokens(text_input, model, tokenizer, device)
    
    # 3. Giải mã và lưu Waveform
    decode_to_wav(audio_tokens, output_wav, device)

if __name__ == "__main__":
    main()