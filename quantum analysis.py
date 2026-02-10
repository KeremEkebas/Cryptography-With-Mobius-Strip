import numpy as np
import matplotlib.pyplot as plt
import secrets
import base64

# Algoritma yükleme
try:
    from endecrypt import SecureMobiusCipher
except ImportError:
    print("❌ HATA: 'endecrypt.py' bulunamadı!")

class PostQuantumDocumentAnalyzer:
    def __init__(self, key_hex):
        self.cipher = SecureMobiusCipher(key_hex)
        # f(0) değerini BV analizi için bir kez hesapla
        self.f0 = self._get_f("00000000") 

    def _get_f(self, input_hex):
        """Nonce etkisinden arındırılmış saf fonksiyon f(x)"""
        enc = self.cipher.encrypt(input_hex)
        return base64.b64decode(enc['ciphertext'])

    def run_deep_analysis(self, iterations=100000):
        print(f"🚀 {iterations} örneklemli analiz başlıyor. Döküman kriterleri uygulanıyor...")
        
        simon_results = []
        bv_violations = []
        s_period = 0xFF # Test edilen gizli periyot s
        
        for i in range(iterations):
            if i % (iterations // 10) == 0:
                print(f"📊 İlerleme: %{int(i/iterations*100)}")

            # 1. SIMON ANALİZİ: f(x) == f(x ⊕ s) denetimi
            x = secrets.randbits(64)
            x_s = x ^ s_period
            
            fx = self._get_f(format(x, '016x'))
            fx_s = self._get_f(format(x_s, '016x'))
            
            # Hamming mesafesi (0'a yakınlık zayıflıktır)
            dist = sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(fx, fx_s))
            simon_results.append(dist / (len(fx) * 8))

            # 2. BERNSTEIN-VAZIRANI: f(a ⊕ b) == f(a) ⊕ f(b) ⊕ f(0) ihlal oranı
            a = secrets.randbits(32)
            b = secrets.randbits(32)
            a_b = a ^ b
            
            fa = self._get_f(format(a, '08x'))
            fb = self._get_f(format(b, '08x'))
            fab = self._get_f(format(a_b, '08x'))
            
            # İhlal kontrolü: fa ^ fb ^ f0 ^ fab
            # Sonuç 0'dan ne kadar uzaksa ihlal o kadar yüksektir (yani güvenlidir)
            violation_score = sum(bin(fa[j] ^ fb[j] ^ self.f0[j] ^ fab[j]).count('1') 
                                 for j in range(len(fab)))
            bv_violations.append(violation_score / (len(fab) * 8))
            
        return simon_results, bv_violations

# Sonuçları Görselleştirme
def plot_doc_results(s_data, b_data):
    plt.figure(figsize=(16, 7))
    
    # Simon Grafiği
    plt.subplot(1, 2, 1)
    plt.hist(s_data, bins=100, color='skyblue', alpha=0.7, density=True)
    plt.axvline(np.mean(s_data), color='black', label=f'Ortalama: {np.mean(s_data):.4f}')
    plt.axvline(0, color='red', linewidth=2, label='Simon Zayıflık (s periyodu)')
    plt.title("Simon Analizi: f(x) vs f(x ⊕ s)\n(Periyot Denetimi)")
    plt.legend()

    # BV Grafiği
    plt.subplot(1, 2, 2)
    plt.hist(b_data, bins=100, color='salmon', alpha=0.7, density=True)
    plt.axvline(np.mean(b_data), color='black', label=f'Ortalama: {np.mean(b_data):.4f}')
    plt.axvline(0, color='red', linewidth=2, label='Tam Lineerlik (BV Zayıflığı)')
    plt.title("BV Analizi: f(a ⊕ b) vs f(a) ⊕ f(b) ⊕ f(0)\n(İhlal Oranı)")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    analyzer = PostQuantumDocumentAnalyzer("3cbaba22914dd09d9a79468e6f2b9a4b22ce5ce28730759f8169313ca69b3615")
    s_results, b_results = analyzer.run_deep_analysis(100000)
    plot_doc_results(s_results, b_results)
