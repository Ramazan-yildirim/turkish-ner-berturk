# BERTurk ile Türkçe Adlandırılmış Varlık Tanıma

Bu proje, Türkçe metinlerde kişi (`PER`), kurum (`ORG`) ve konum (`LOC`) varlıklarını bulmak için BERTurk modelini WikiANN veri kümesi üzerinde ince ayarlar. Eğitim, değerlendirme, en iyi modelin saklanması ve örnek veri incelemesi tek bir Jupyter notebook içinde uçtan uca gösterilir.

## Proje özeti

- Temel model: `dbmdz/bert-base-turkish-cased`
- Görev: token classification / Named Entity Recognition (NER)
- Veri kümesi: `unimelb-nlp/wikiann`, Türkçe (`tr`) yapılandırması
- Etiket şeması: BIO
- Değerlendirme: `seqeval` ile precision, recall, F1 ve token accuracy
- En iyi model seçimi: doğrulama F1 skoru
- Kayıt politikası: test F1 önceki kaydı aşarsa model ve metrikler güncellenir

## Etiketler

| ID | Etiket | Anlamı |
|---:|---|---|
| 0 | `O` | Varlık değil |
| 1 | `B-PER` | Kişi adının başlangıcı |
| 2 | `I-PER` | Kişi adının devamı |
| 3 | `B-ORG` | Kurum adının başlangıcı |
| 4 | `I-ORG` | Kurum adının devamı |
| 5 | `B-LOC` | Konum adının başlangıcı |
| 6 | `I-LOC` | Konum adının devamı |

Notebook'un etiket keşfi çıktısı:

```text
Etiketler: ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
```

## Proje yapısı

```text
turkish-ner-berturk/
├── prepare_data.ipynb   # Veri hazırlama, eğitim, değerlendirme ve model kaydı
├── requirements.txt     # Doğrulanmış doğrudan Python bağımlılıkları
├── .gitignore           # Ortam, önbellek, checkpoint ve model çıktıları
└── README.md
```

Eğitim sırasında `results/` altında checkpoint'ler, başarılı model kaydında ise `saved_berturk_ner/` altında model, tokenizer ve `best_metrics.json` üretilir. Bu dosyalar yüzlerce MB büyüklüğe ulaşabildiği için Git tarafından izlenmez. Mevcut çalışmada `results/` yaklaşık 3,69 GB, dışa aktarılan model ise yaklaşık 420,49 MB'tır.

## Kurulum

Python 3.12 önerilir. Aşağıdaki sürümler Python 3.12.2 ile doğrulanmıştır.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name turkish-ner-berturk --display-name 'Turkish NER (BERTurk)'
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name turkish-ner-berturk --display-name 'Turkish NER (BERTurk)'
```

GPU ile eğitim yapılacaksa önce sistemdeki CUDA sürümüne uygun PyTorch paketi [PyTorch kurulum seçicisinden](https://pytorch.org/get-started/locally/) kurulmalıdır. Notebook CUDA kullanılabilirliğini otomatik olarak kontrol eder; CUDA yoksa eğitim CPU üzerinde çalışır ve belirgin ölçüde daha uzun sürer.

İlk çalıştırmada WikiANN veri kümesi, BERTurk ağırlıkları ve `seqeval` değerlendirme modülü Hugging Face üzerinden indirilir; bu nedenle internet bağlantısı gerekir.

## Çalıştırma

1. `prepare_data.ipynb` dosyasını VS Code veya tercih edilen Jupyter arayüzünde açın.
2. Kernel olarak `Turkish NER (BERTurk)` ortamını seçin.
3. Hücreleri yukarıdan aşağıya sırayla çalıştırın.
4. Eğitim tamamlanınca test metriklerini ve `saved_berturk_ner/` klasörünü kontrol edin.

Notebook'un temel veri yükleme ve tokenizer kurulumu şöyledir:

```python
from datasets import load_dataset
from transformers import AutoTokenizer

dataset = load_dataset('unimelb-nlp/wikiann', 'tr')
model_name = 'dbmdz/bert-base-turkish-cased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

Doğrulanmış veri bölümleri:

```text
DatasetDict({
    validation: Dataset({ num_rows: 10000 })
    test:       Dataset({ num_rows: 10000 })
    train:      Dataset({ num_rows: 20000 })
})
```

## Token-etiket hizalama

BERT tokenizer bir kelimeyi birden fazla alt parçaya ayırabilir. Veri kümesindeki kelime düzeyi etiketi yalnızca ilk alt parçaya atanır; özel token'lar ile devam alt parçaları kayıp hesabından çıkarılmak üzere `-100` alır.

```python
def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        is_split_into_words=True,
    )

    labels = []
    for batch_index, word_labels in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_index)
        previous_word_id = None
        label_ids = []

        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            elif word_id != previous_word_id:
                label_ids.append(word_labels[word_id])
            else:
                label_ids.append(-100)
            previous_word_id = word_id

        labels.append(label_ids)

    tokenized_inputs['labels'] = labels
    return tokenized_inputs
```

Notebook'taki ilk örnekten kısaltılmış hizalama çıktısı:

```text
TOKEN                | ETİKET ID  | ETİKET ADI
--------------------------------------------------
[CLS]                | -100       | MASKE -100
3                    | 0          | O
.                    | -100       | MASKE -100
lük                  | -100       | MASKE -100
maçında              | 0          | O
Slovenya             | 3          | B-ORG
Millî                | 4          | I-ORG
Basketbol            | 4          | I-ORG
Takımı               | 4          | I-ORG
[SEP]                | -100       | MASKE -100
```

## Eğitim yapılandırması

| Parametre | Değer |
|---|---:|
| Epoch | 3 |
| Öğrenme oranı | `2e-5` |
| Eğitim batch boyutu | 16 |
| Değerlendirme batch boyutu | 16 |
| Weight decay | `0.01` |
| Değerlendirme sıklığı | Her epoch |
| Checkpoint sıklığı | Her epoch |
| En iyi model metriği | F1 |

```python
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy='epoch',
    save_strategy='epoch',
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model='f1',
    logging_steps=100,
)
```

## Sonuçlar

Kayıtlı checkpoint geçmişindeki doğrulama sonuçları:

| Epoch | Validation loss | Precision | Recall | F1 | Accuracy |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.1200 | 0.8958 | 0.9105 | 0.9031 | 0.9659 |
| 2 | 0.1148 | 0.9132 | 0.9239 | 0.9185 | 0.9699 |
| 3 | 0.1198 | 0.9132 | 0.9282 | **0.9207** | 0.9712 |

`load_best_model_at_end=True` sayesinde değerlendirme sonunda doğrulama F1'ı en yüksek checkpoint kullanılır. Notebook'ta kaydedilmiş test sonucu:

```text
--- TEST SONUÇLARI ---
Precision : 0.9134
Recall    : 0.9244
F1 Score  : 0.9189
Accuracy  : 0.9714
```

Ham değerler `saved_berturk_ner/best_metrics.json` dosyasına yazılır. Sonuçlar mevcut yerel eğitim çalışmasına aittir; donanım, rastgelelik, paket sürümleri ve veri/model güncellemeleri yeni çalışmalarda küçük farklar oluşturabilir.

## Eğitilmiş modelle tahmin

Notebook başarıyla tamamlandıktan ve `saved_berturk_ner/` oluşturulduktan sonra model şu şekilde kullanılabilir:

```python
from transformers import pipeline

ner = pipeline(
    'ner',
    model='./saved_berturk_ner',
    tokenizer='./saved_berturk_ner',
    aggregation_strategy='simple',
    device=-1,  # CPU; ilk CUDA aygıtı için 0 kullanın
)

entities = ner(
    '''Mustafa Kemal Atatürk Ankara'da Türkiye Büyük Millet Meclisini ziyaret etti.'''
)

for entity in entities:
    print(
        entity['entity_group'],
        entity['word'],
        '{:.4f}'.format(float(entity['score'])),
    )
```

Kayıtlı modelle CPU üzerinde doğrulanmış çıktı:

```text
PER Mustafa Kemal Atatürk 0.9908
LOC Ankara 0.9964
ORG Türkiye Büyük Millet Meclisi 0.9985
```

## Üretilen dosyalar

`results/checkpoint-*` klasörleri modelin yanı sıra optimizer ve scheduler durumlarını da sakladığı için oldukça büyüktür. `saved_berturk_ner/` ise tahminde gereken son modeli ve tokenizer'ı içerir:

```text
saved_berturk_ner/
├── best_metrics.json
├── config.json
├── model.safetensors
├── tokenizer.json
├── tokenizer_config.json
└── training_args.bin
```

Bu klasörleri paylaşmak için Git deposuna eklemek yerine Hugging Face Hub, bir model kayıt sistemi veya sürümlü nesne depolama kullanılması önerilir.
