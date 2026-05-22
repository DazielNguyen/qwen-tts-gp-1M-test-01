import os
import io
import sys
import json
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset, Audio
from transformers import AutoTokenizer
from tqdm import tqdm

# --- KẾT NỐI VỚI MÃ NGUỒN QWEN LOCAL ---
QWEN_REPO_PATH = os.path.join(os.getcwd(), "Qwen3-TTS")
if not os.path.exists(QWEN_REPO_PATH):
    print(f"❌ Lỗi: Không tìm thấy thư mục {QWEN_REPO_PATH}!")
    print("Vui lòng chạy lệnh sau ngoài Terminal trước: git clone https://github.com/QwenLM/Qwen3-TTS.git")
    sys.exit(1)
    
sys.path.append(QWEN_REPO_PATH)

# Import chính xác class từ file bạn vừa gửi
from qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 import Qwen3TTSTokenizerV2Model

def main():
    os.makedirs("data", exist_ok=True) 
    output_file = "data/processed_dataset_12hz.jsonl"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị xử lý: {device.upper()}")

    print("1. Đang tải Qwen3 Text Tokenizer từ Hugging Face...")
    text_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base", trust_remote_code=True)

    print("2. Đang khởi tạo Qwen3 Audio Tokenizer (12Hz) bằng Local Code...")
    # Tải trọng số từ HF nhưng dựng khung bằng Class Local bạn gửi để tránh KeyError
    audio_tokenizer = Qwen3TTSTokenizerV2Model.from_pretrained("Qwen/Qwen3-TTS-Tokenizer-12Hz").to(device)
    audio_tokenizer.eval()

    print("3. Đang kết nối tới LibriSpeech (Streaming)...")
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    # Đọc cấu hình chuẩn mẫu từ model Qwen3 để tự động hóa Hz
    target_sr = audio_tokenizer.get_input_sample_rate() 
    print(f"-> Tần số lấy mẫu chuẩn của Qwen3: {target_sr} Hz")
    
    total_files = 28539 
    print(f"4. Bắt đầu trích xuất hệ thống 12Hz - 16 Codebooks...")
    
    with open(output_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(dataset, desc="Tiến độ", total=total_files, unit="file")):
            # --- BƯỚC 1: XỬ LÝ TEXT ---
            text = item["text"]
            text_tokens = text_tokenizer.encode(text) 
            
            # --- BƯỚC 2: XỬ LÝ AUDIO ---
            audio_bytes = item["audio"]["bytes"]
            audio_array, original_sr = sf.read(io.BytesIO(audio_bytes))
            audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
            
            if original_sr != target_sr:
                if len(audio_tensor.shape) == 1:
                    audio_tensor = audio_tensor.unsqueeze(0)
                resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
                audio_tensor = resampler(audio_tensor).squeeze(0)
            
            # Đưa về dạng [Batch=1, Seq_len] theo đúng thiết kế hàm encode của Qwen3 V2
            audio_input = audio_tensor.unsqueeze(0).to(device)
            # Tạo padding mask toàn bộ là 1 (không mask) tương ứng với độ dài audio
            padding_mask = torch.ones_like(audio_input, dtype=torch.long, device=device)
            
            with torch.no_grad():
                # Trích xuất ra Object Output chuẩn mã nguồn
                encoded_output = audio_tokenizer.encode(input_values=audio_input, padding_mask=padding_mask)
                
                # Biến đổi cấu trúc: output đầu ra là một List các tensor cho từng batch.
                # Mỗi tensor có shape: [num_quantizers, codes_length]
                # Ta bốc batch_0 và giới hạn lấy đúng 16 tầng codebook đầu tiên
                audio_16_codebooks = encoded_output.audio_codes[0][:16, :]
            
            # --- BƯỚC 3: ĐÓNG GÓI JSONL ---
            sample = {
                "id": i,
                "raw_text": text,
                "text_tokens": text_tokens,
                "audio_codes": audio_16_codebooks.cpu().tolist()
            }
            
            f.write(json.dumps(sample) + "\n")

    print(f"\n✅ Hoàn tất! Toàn bộ dữ liệu 12Hz đã đóng gói vào: {output_file}")

if __name__ == "__main__":
    main()