# MLST_New — FRPV-PH fibrozis çalışması (Konsept A + B)

Uzman farkını modelleyen, iç içe (fibrozis ⊂ duvar) yapıyı garanti eden, Beer–Lambert boya fiziğiyle
güçlendirilmiş segmentasyon + fibrozis indeksi (FI) için hayvan düzeyinde conformal aralıklar +
tedavi etkisi için prediction-powered inference (PPI).
Tasarımın tamamı: `docs/STUDY_DESIGN.md`.

Hedef klasör: `C:\Users\Workstation\Desktop\Makale\MLST_New`

---

## 1. Klasör yapısı

```
MLST_New/
├─ RUN_1_setup_and_smoke.bat        # ortam kontrolü + veri hazırlığı + testler + kısa GPU denemesi
├─ RUN_2_backbones.bat              # Aşama 1: omurga karşılaştırması (uzun)
├─ RUN_3_ablations_and_analysis.bat # Aşama 2-4 + baseline'lar + tüm istatistikler
├─ environment.yml / requirements*.txt
├─ configs/base.yaml                # TÜM ayarlar (yorumlu)
├─ configs/exp/main.yaml            # önerilen yöntem
├─ plans/stage*.yaml                # deney planları (kaldığı yerden devam eder)
├─ data/
│   ├─ splits_5fold_official.csv    # makalenin resmi hayvan-düzeyi fold'ları
│   └─ FRPV-PH/                     # <-- VERİ SETİNİ BURAYA AÇIN (raw_images, roi_masks, expert_1, expert_2)
├─ cache/                           # otomatik: .npy önbellek, meta.csv, stain_vectors.json
├─ results/<deney_adı>/             # otomatik: fold*/model, predictions.csv, oof_predictions.csv, analizler
├─ src/frpv/                        # kütüphane (data, stain, models, eval, uq, stats)
├─ scripts/                         # 00..07 adım betikleri
├─ baselines/                       # nnU-Net, yayınlanan UNet (düzeltilmiş), klasik renk ayrıştırma
├─ tests/test_models.py
└─ docs/STUDY_DESIGN.md
```

## 2. Kurulum (bir kez)

**Yeni conda ortamı (önerilen):**
```bat
cd C:\Users\Workstation\Desktop\Makale\MLST_New
conda env create -f environment.yml
conda activate mlst
pip install -r requirements-baselines.txt     :: nnU-Net + MONAI (baseline'lar için)
```

**Mevcut bir ortamı kullanacaksanız:** önce CUDA'lı torch kurun, sonra:
```bat
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt -r requirements-baselines.txt
```
(cu128 tekerlekleri RTX 50xx dahil tüm güncel NVIDIA kartlarda çalışır.)

**Patoloji temel modelleri (gated):** huggingface.co'da şu sayfalarda lisansı kabul edin, sonra
`huggingface-cli login` ile token girin:
MahmoodLab/UNI2-h · paige-ai/Virchow2 · bioptimus/H-optimus-1 (ve isteğe bağlı MahmoodLab/uni,
bioptimus/H-optimus-0, owkin/phikon-v2, histai/hibou-L).
Erişim yoksa `plans/stage1_backbones.yaml` içinden o satırı silin; DINOv2-L ve ConvNeXt açık modellerdir.

## 3. Veri

Figshare'den indirdiğiniz arşivi `data\FRPV-PH\` altına açın. Betik `raw_images` klasörünü alt
dizinlerde kendisi bulur. Başka bir yerdeyse: `python scripts\01_prepare_data.py --data_root D:\...`
ya da `configs/base.yaml > paths.data_root`.
(GitHub reposundaki `dataset/` klasörü de aynı veriyi içerir.)

## 4. Çalıştırma

```bat
conda activate mlst
cd C:\Users\Workstation\Desktop\Makale\MLST_New
RUN_1_setup_and_smoke.bat            :: ~15-30 dk
RUN_2_backbones.bat                  :: Aşama 1 (5 omurga x 5 fold)
RUN_3_ablations_and_analysis.bat     :: en iyi omurgayı İÇ-DOĞRULAMA metriklerine göre seçer
baselines\nnunet\run_nnunet.bat      :: (ayrı, çok uzun; paralel GPU varsa eşzamanlı)
```
Tüm adımlar **kaldığı yerden devam eder**: bir fold'un `predictions.csv` dosyası varsa atlanır.
Hata alan koşu `results\plan_failures.log`'a yazılır, plan devam eder.

Tek deney / tek fold denemek:
```bat
python scripts\02_train_cv.py --exp configs\exp\main.yaml --folds 0 model.encoder=fm:uni2h exp_name=test_uni2h
```
Her ayar komut satırından `anahtar=değer` ile değiştirilebilir (ör. `train.batch_size=4`).

## 5. Tahmini süre ve bellek

| Omurga | VRAM (batch 8, bf16) | ~süre / fold* |
|---|---|---|
| ConvNeXt-S | ~10 GB | 20-30 dk |
| DINOv2-L / Virchow2 / H-optimus-1 (ViT-L/G) | 14-24 GB | 40-90 dk |
| UNI2-h (ViT-H) | ~20-28 GB | 60-100 dk |

\*60 epoch × 2000 kırpıntı; RTX 4090/5090 sınıfı. Bellek yetmezse `train.batch_size=4`.
Toplam: Aşama 1 ≈ 1-2 gün, Aşama 2 ≈ 2-3 gün, Aşama 3-4 ≈ 1 gün, nnU-Net ≈ 1-2 gün.
Önce `RUN_1` sonuçlarını kontrol edip devam etmenizi öneririm.

## 6. Çıktılar (makale için bana lazım olanlar)

`RUN_3` sonunda **`results_for_paper.zip`** oluşur (yalnızca csv/json/yaml; model ağırlıkları yok).
Bunu bana göndermeniz yeterli. İçerik (`<tag>` = omurga, ör. `uni2h`):
```
results/summary/summary.csv, summary.md, human_reference.csv, paired_vs_a2_dual_<tag>.csv
results/<deney>/oof_predictions.csv                 # her görüntü için tüm metrikler
results/<deney>/fold*/train_log.csv, stain_vectors.json, stain_matrix_learned.json
results/final_ens_<tag>/conformal/grid.csv          # conformal kapsama/genişlik tablosu
results/final_ens_<tag>/ppi/{contrasts,null,budget}.csv
results/final_ens_<tag>/rater/{dependence,bias_by_project}.csv, image_bias.json
results/experts_only/rater/...                      # motivasyon analizi (model gerektirmez)
cache/data_report.json, cache/stain_vectors.json, cache/meta.csv
```
Aşama 2-4 koşu adlarına omurga etiketi eklenir (ör. `a2_dual_uni2h`), böylece farklı omurgalarla
yapılan koşular birbirinin üzerine yazılmaz.
Checkpoint'ler yalnızca eğitilen ağırlıkları (LoRA, dekoder, başlıklar) içerir; dondurulmuş temel model
ağırlıkları yükleme sırasında HF önbelleğinden okunur (disk dolmaz). Şekiller için
`configs/base.yaml > infer.save_probmaps: true` ile final modeli yeniden çalıştırmak yeterli olacak
(şekil betiklerini sonuçları görünce birlikte yazacağız).

## 7. Sorun giderme

* `CUDA out of memory` → `train.batch_size=4` (gerekirse `data.crop_size=640`).
* DataLoader takılması (Windows) → `train.num_workers=0` ile deneyin.
* `401 / gated repo` → HF lisansı + `huggingface-cli login`.
* `torch.compile` Windows'ta kapalı tutulur (`train.compile: false`).
* nnU-Net: `nnUNet_compile=f` bat dosyasında ayarlı; daha hızlı koşu için `-tr nnUNetTrainer_250epochs`.
