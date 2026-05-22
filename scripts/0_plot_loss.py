import re
import matplotlib.pyplot as plt
import os

def plot_loss_from_log(log_file, output_image):
    if not os.path.exists(log_file):
        print(f"Không tìm thấy file {log_file}!")
        return

    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    epochs = []
    losses = []

    # Dùng Regex để tìm các dòng chứa "Hoàn tất Epoch X | Avg Loss: Y"
    pattern = re.compile(r"Epoch (\d+)\s*\|\s*Avg Loss:\s*([\d.]+)")

    for line in lines:
        match = pattern.search(line)
        if match:
            epochs.append(int(match.group(1)))
            losses.append(float(match.group(2)))

    if not epochs:
        print("Không tìm thấy dữ liệu Loss nào trong file log. Bạn copy đúng chưa?")
        return

    # Vẽ đồ thị
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, losses, marker='o', linestyle='-', color='b', linewidth=2, markersize=6)
    
    plt.title('Training Loss Giai đoạn 1 (Auto-Regressive)', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Lưu thành file ảnh
    plt.savefig(output_image, dpi=300)
    print(f"Đã vẽ xong! Hãy mở file {output_image} để xem đồ thị nhé.")

if __name__ == "__main__":
    plot_loss_from_log("ar_train_log.txt", "ar_loss_curve.png")