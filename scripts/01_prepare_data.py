import os
import io
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset, Audio
from transformers import EncodecModel, AutoProcessor, AutoTokenizer
from tqdm import tqdm

def main():
    # Tạo thư mục mới để không bị lẫn với dữ liệu cũ
    os.makedirs("data/tokens_16cb", exist_ok=True) 
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị xử lý: {device.upper()}")

    print("1. Đang tải BPE Tokenizer (GPT-2)...")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    print("2. Đang tải mô hình EnCodec...")
    processor = AutoProcessor.from_pretrained("facebook/encodec_24khz")
    model = EncodecModel.from_pretrained("facebook/encodec_24khz").to(device)
    model.eval()

    print("3. Đang kết nối tới LibriSpeech (Streaming - Bypass Lỗi)...")
    dataset = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    dataset = dataset.cast_column("audio", Audio(decode=False))

    target_sr = 24000
    total_files = 28539 
    
    print(f"4. Bắt đầu trích xuất 16 Codebooks cho {total_files} mẫu...")
    
    for i, item in enumerate(tqdm(dataset, desc="Tiến độ", total=total_files, unit="file")):
        # -----------------------------------------
        # BƯỚC 1: XỬ LÝ VĂN BẢN (TEXT TOKENS)
        # -----------------------------------------
        text = item["text"]
        # Mã hóa BPE. tensor trả về có dạng [1, seq_len], dùng squeeze để ép về 1D [seq_len]
        text_tokens = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)
        
        # -----------------------------------------
        # BƯỚC 2: XỬ LÝ ÂM THANH (AUDIO CODES)
        # -----------------------------------------
        audio_bytes = item["audio"]["bytes"]
        audio_array, original_sr = sf.read(io.BytesIO(audio_bytes))
        
        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
        
        if original_sr != target_sr:
            if len(audio_tensor.shape) == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            resampler = torchaudio.transforms.Resample(orig_freq=original_sr, new_freq=target_sr)
            audio_tensor = resampler(audio_tensor).squeeze(0)
        
        audio_ready = audio_tensor.numpy()

        inputs = processor(raw_audio=audio_ready, sampling_rate=24000, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # Ép băng thông 12.0 để EnCodec xuất ra chính xác 16 Codebooks
            audio_codes = model.encode(inputs["input_values"], inputs["padding_mask"], bandwidth=12.0).audio_codes
        
        # Trích xuất toàn bộ 16 layers.
        # Shape gốc: [batch=1, num_quantizers=16, channels=1, time_steps]
        # Sau khi trích xuất: [16, time_steps]
        audio_16_codebooks = audio_codes[0, :16, 0, :].cpu()
        
        # -----------------------------------------
        # BƯỚC 3: LƯU TRỮ CHUẨN QWEN-TTS
        # -----------------------------------------
        save_path = f"data/tokens_16cb/sample_{i}.pt"
        torch.save({
            "text_tokens": text_tokens,
            "audio_codes": audio_16_codebooks,
            "raw_text": text  # Lưu kèm text gốc để tiện debug sau này
        }, save_path)

    print("\nHoàn tất việc chuẩn bị dữ liệu chuẩn 16 Codebooks!")

if __name__ == "__main__":
    main()