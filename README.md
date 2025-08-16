#  DüğünBuketi AI Sohbet Analiz Sistemi

Bu proje, DüğünBuketi platformundan alınan müşteri konuşmalarını yapay zeka ile analiz ederek yanıtlanmamış soruları tespit eder, duygu analizi yapar ve müşteri hizmetleri kalitesini ölçer.

## Özellikler

###  LLM Entegrasyonu
- **OpenAI GPT-4o** - En yeni model 
- **OpenAI GPT-4** ve **GPT-3.5-turbo**
- **Groq** (Llama3, Mixtral) - Ücretsiz 
- **Hugging Face** - Ücretsiz
- **Anthropic Claude**

###  Analiz Çıktıları
- **Bot Yanıt Verdi Mi?** → Evet / Hayır (Müşteri hizmetleri kalitesi)
- **Sentiment** → Pozitif / Negatif / Nötr
- **Konu Kategorisi** → Düğün mekanı, Gelinlik, Fotoğrafçı, Müzik/DJ, vb.
- **Detaylı İstatistikler** → Token kullanımı, API çağrısı sayısı

###  Manuel Etiketleme Arayüzü
- **Streamlit** tabanlı web arayüzü
- LLM tahminlerini manuel doğrulama
- Gerçek zamanlı doğruluk hesaplama
- Görsel performans raporları

###  Doğruluk Analizi
- Confusion Matrix görselleştirme
- Precision, Recall, F1-Score metrikleri
- Hedef başarı oranı kontrolü (%85 hedef)
- Detaylı performans raporları

## Kurulum

### 1. Gereksinimler
```bash
pip install -r requirements.txt
```

### 2. API Anahtarları
`.env` dosyası oluşturun:
```env
# OpenAI (Önerilen)
OPENAI_API_KEY=your_openai_api_key_here

# Alternatif Sağlayıcılar
ANTHROPIC_API_KEY=your_anthropic_key
GROQ_API_KEY=your_groq_key
HUGGINGFACE_API_KEY=your_hf_key

# Varsayılan Ayarlar
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4o
MAX_RETRIES=3
API_TIMEOUT=30
```

### 3. NLTK Verilerini İndirin
```python
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
```

## Kullanım

### Ana Workflow
```bash
python main_workflow.py
```

**Adım 1: LLM Analizi**
1. JSON chat dosyasını seçin
2. LLM provider/model seçin (GPT-4o önerilen)
3. Otomatik analiz başlatın

**Adım 2: Manuel Etiketleme**
```bash
streamlit run enhanced_manual_labeling.py
```
- Web arayüzünde LLM tahminlerini doğrulayın
- Manuel etiketli dosyayı indirin

**Adım 3: Doğruluk Analizi**
- Manuel etiketli dosyayı yükleyin
- Detaylı performans raporu alın

### Hızlı Test
```bash
# Test verisi oluştur
python test_data_generator.py

# Temel analiz
python main.py
```

##  Çıktı Formatları

### LLM Analiz Sonuçları
```csv
message_id,sender,message,sentiment_llm,topic_llm,bot_response_llm,confidence_sentiment,confidence_topic,confidence_bot_response
1,müşteri_1,"Düğün mekanı arıyorum",Nötr,Düğün mekanı,Hayır,0.95,0.92,0.88
```

### Manuel Etiketli Veri
```csv
message_id,sentiment_llm,sentiment_manual,topic_llm,topic_manual,bot_response_llm,bot_response_manual,accuracy_sentiment,accuracy_topic,accuracy_bot_response
```

### Doğruluk Metrikleri
```json
{
  "sentiment": {
    "accuracy": 0.92,
    "precision": 0.91,
    "recall": 0.93,
    "f1_score": 0.92
  },
  "topic": {
    "accuracy": 0.88,
    "precision": 0.87,
    "recall": 0.89,
    "f1_score": 0.88
  },
  "bot_response": {
    "accuracy": 0.95,
    "precision": 0.94,
    "recall": 0.96,
    "f1_score": 0.95
  }
}
```

## Düğün Buketi Kategorileri

### Konu Kategorileri
-  **Düğün mekanı** - Salon, bahçe, kır düğünü
-  **Gelinlik** - Gelinlik modelleri, kiralama
-  **Fotoğrafçı** - Düğün fotoğrafçısı, video çekimi
-  **Müzik/DJ** - Müzik grubu, DJ hizmeti
-  **Çiçek/Dekorasyon** - Gelin buketi, masa süsleme
-  **Davetiye** - Düğün davetiyesi tasarımı
-  **Pasta/Catering** - Düğün pastası, yemek servisi
-  **Takı/Aksesuar** - Düğün takıları, aksesuar
-  **Nikah şekeri** - Nikah şekeri, hediyeler
-  **Genel bilgi** - Genel sorular
-  **Fiyat sorgusu** - Fiyat bilgileri
-  **Rezervasyon** - Randevu, rezervasyon

### Sentiment Kategorileri
-  **Pozitif** - Memnuniyet, teşekkür
-  **Nötr** - Bilgi talebi, soru
-  **Negatif** - Şikayet, memnuniyetsizlik

### Bot Yanıt Durumu
-  **Evet** - Müşteri mesajına yanıt verildi
-  **Hayır** - Müşteri mesajı yanıtsız kaldı

