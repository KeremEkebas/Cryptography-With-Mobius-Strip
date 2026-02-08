import numpy as np
import matplotlib.pyplot as plt
import secrets
import base64

# Algoritma yükleme
try:
    from endecrypt import SecureMobiusCipher
except ImportError:
    print("❌ HATA: 'endecrypt.py' bulunamadı!")

class AvalancheAnalyzer:
    def __init__(self, key_hex):
        self.cipher = SecureMobiusCipher(key_hex)

    def _get_f(self, input_hex):
        """Nonce etkisinden arındırılmış saf fonksiyon f(x)"""
        # Not: Nonce her seferinde aynı kalmalı ki değişim sadece girdiden kaynaklansın
        # Eğer cipher sınıfın içerde otomatik nonce üretiyorsa, bu testi yanıltabilir.
        enc = self.cipher.encrypt(input_hex)
        return base64.b64decode(enc['ciphertext'])

    def run_avalanche_test(self, iterations=50000, bit_length=64):
        print(f"🚀 {iterations} örneklemli Avalanche Effect analizi başlıyor...")
        
        diff_ratios = []

        for i in range(iterations):
            if i % (iterations // 10) == 0:
                print(f"📊 İlerleme: %{int(i/iterations*100)}")

            # 1. Orijinal girdi (x)
            x_val = secrets.randbits(bit_length)
            x_hex = format(x_val, f'0{bit_length//4}x')
            
            # 2. Sadece 1 biti değiştirilmiş girdi (x')
            flip_bit = secrets.randbelow(bit_length)
            x_prime_val = x_val ^ (1 << flip_bit)
            x_prime_hex = format(x_prime_val, f'0{bit_length//4}x')

            # Şifrele
            fx = self._get_f(x_hex)
            fx_prime = self._get_f(x_prime_hex)

            # Hamming Mesafesi Hesapla (Kaç bit değişti?)
            # f(x) ve f(x') arasındaki XOR'lanmış bitlerin toplamı
            diff_bits = sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(fx, fx_prime))
            total_bits = len(fx) * 8
            
            diff_ratios.append(diff_bits / total_bits)

        return diff_ratios

def plot_avalanche_results(data):
    mean_val = np.mean(data)
    std_val = np.std(data)
    
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.hist(data, bins=100, color='mediumpurple', alpha=0.7, edgecolor='black', density=True)
    
    # İdeal %50 çizgisi
    plt.axvline(0.5, color='red', linestyle='--', linewidth=2, label='İdeal Avalanche (%50)')
    # Gerçek Ortalama
    plt.axvline(mean_val, color='green', linewidth=2, label=f'Hesaplanan Ortalama: %{mean_val*100:.2f}')
    
    plt.title("Avalanche Effect (Çığ Etkisi) Analizi")
    plt.xlabel("Değişen Bit Oranı (Hamming Distance / Total Bits)")
    plt.ylabel("Yoğunluk")
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    
    # İstatistiksel Not
    plt.figtext(0.7, 0.5, f"Ortalama: {mean_val:.4f}\nStd Sapma: {std_val:.4f}\nMin: {min(data):.4f}\nMax: {max(data):.4f}", 
                bbox=dict(facecolor='white', alpha=0.5))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Test anahtarı
    key = "3cbaba22914dd09d9a79468e6f2b9a4b22ce5ce28730759f8169313ca69b3615"
    
    analyzer = AvalancheAnalyzer(key)
    results = analyzer.run_avalanche_test(iterations=50000)
    plot_avalanche_results(results) avalanche
