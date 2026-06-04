# Yazılım Geliştirme Dersi — 2. Proje

Zaman serisi anomali tespitinde black-box (LSTM, GRU) ve yorumlanabilir (Probabilistic Automata) modellerin karşılaştırmalı analizi.

---

## Veri Setleri

### SKAB (Skoltech Anomaly Benchmark)

| Özellik | Değer |
|---------|-------|
| Kullanılan klasörler | `valve1`, `valve2` |
| Toplam satır | 22.472 |
| Anomali oranı | %34.8 |
| Hedef değişken | `anomaly` |
| Model girdisi | 8 sensör değişkeni |
| Değerlendirme | StratifiedGroupKFold (k=5, grup: `source_file`) |

### BATADAL (Battle of the Attack Detection Algorithms)

| Özellik | Değer |
|---------|-------|
| Kullanılan dosya | `BATADAL_dataset04.csv` (Training Dataset 2) |
| Toplam satır | 4.177 |
| Saldırı oranı | %5.2 |
| Hedef değişken | `ATT_FLAG` (−999→0 normal, 1→1 saldırı) |
| Model girdisi | 43 sensör/sistem değişkeni |
| Değerlendirme | Zaman sıralı %60 / %20 / %20 |

---

**Çıktı dosyaları:**

```
results/
├── skab_results.json           ← SKAB metrikleri (mean ± std)
├── batadal_results.json        ← BATADAL metrikleri
├── skab_sweep.json             ← Parametre taraması
├── batadal_sweep.json
├── skab_explanations.json      ← Açıklanabilirlik örnekleri
├── batadal_explanations.json
├── skab_stats.json             ← Wilcoxon testi
├── batadal_stats.json
```

---

## Yazılım Mimarisi

```
yazlab-2/
├── config.py                  # Merkezi konfigürasyon — tüm parametreler
├── run_all.py                 # Ana çalıştırıcı
├── utils.py                   # Metrik, loglama, görselleştirme
├── data_proc/
│   ├── loader.py              # SKAB + BATADAL yükleme
│   └── preprocessor.py        # Scaler, PCA, bölme, gürültü
├── models/
│   ├── dl_models.py           # LSTM, GRU (kaydetme/yükleme dahil)
│   └── automata_model.py      # PAA → SAX → Probabilistic Automata
├── explainability/
│   └── explainer.py           # Path prob., güven skoru, counterfactual
├── experiments/
│   ├── batadal_exp.py
│   └── skab_exp.py
├── tests/
│   └── test_unseen.py         # Testler
├── models_saved/              # Eğitilmiş modeller (.keras, .pkl)
├── results/                   # JSON çıktılar
└── plots/                     # PNG görseller
```

---

**Ortak eğitim parametreleri:**

| Parametre | Değer |
|-----------|-------|
| Epoch üst sınırı | 50 |
| Batch size | 32 |
| Early stopping | `val_loss`, patience=5 |
| Optimizör | Adam (lr=0.001) |
| Sınıf ağırlığı | `{0: 1.0, 1: max(neg/pos, 5.0)}` |
| Eşik seçimi | Val seti F1 maksimizasyonu |
| Random seed'ler | 42, 123, 2026, 7, 999 |

---

## Deneysel Tasarım

| Senaryo | Açıklama |
|---------|----------|
| **Orijinal** | Ham normalleştirilmiş veri |
| **Gürültülü** | σ=0.05 Gaussian gürültü eklenmiş |
| **Unseen** | Test setinin son %10'u |

**Sabit Automata parametreleri:** window_size=4, alphabet_size=3

**Parametre varyasyonu:** window_size ∈ {3,4,5,6}, alphabet_size ∈ {3,4,5,6}
