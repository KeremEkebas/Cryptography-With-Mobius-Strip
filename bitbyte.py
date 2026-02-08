import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import os

def analyze_endecrypt_file(file_path='endecrypt'):
    # 1. Dosya Kontrolü ve Okuma
    if not os.path.exists(file_path):
        print(f"❌ Hata: '{file_path}' dosyası bulunamadı! Lütfen dosya adını kontrol edin.")
        return

    with open(file_path, "rb") as f:
        data = f.read()

    file_size = len(data)
    if file_size == 0:
        print("⚠️ Dosya boş, analiz yapılamıyor.")
        return

    print(f"✅ Dosya okundu: {file_path}")
    print(f"📏 Boyut: {file_size} byte ({file_size / 1024:.2f} KB)")

    # 2. Byte Dağılımı Hesaplama (0-255)
    byte_counts = Counter(data)
    frequencies = [byte_counts.get(i, 0) for i in range(256)]
    ideal_line = file_size / 256

    # 3. Bit Dağılımı Hesaplama
    total_bits = file_size * 8
    ones = sum(bin(byte).count('1') for byte in data)
    zeros = total_bits - ones
    
    one_percent = (ones / total_bits) * 100
    zero_percent = (zeros / total_bits) * 100

    # 4. Görselleştirme
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Sol Grafik: Byte Değeri Dağılımı ---
    ax1.bar(range(256), frequencies, color='#3b82f6', width=1.0, alpha=0.8)
    ax1.axhline(y=ideal_line, color='red', linestyle='-', label=f'İdeal: {ideal_line:.2f}')
    ax1.set_title(f"Byte Değeri Dağılımı (0-255)\n{file_path}", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Byte Değeri")
    ax1.set_ylabel("Frekans")
    ax1.legend()
    ax1.grid(axis='y', linestyle='--', alpha=0.4)

    # --- Sağ Grafik: Bit Dağılım Oranı ---
    bars = ax2.bar(['Bit 0', 'Bit 1'], [zeros, ones], color=['#f87171', '#60a5fa'])
    ax2.set_title("Bit Dağılım Oranı (%0 ve %1)", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Toplam Bit Sayısı")
    
    # Yüzde etiketlerini barların ortasına yaz
    ax2.text(0, zeros/2, f'%{zero_percent:.2f}', ha='center', color='white', fontweight='bold', fontsize=13)
    ax2.text(1, ones/2, f'%{one_percent:.2f}', ha='center', color='white', fontweight='bold', fontsize=13)

    plt.tight_layout()
    plt.show()

# Analizi Başlat
analyze_endecrypt_file('endecrypt')
