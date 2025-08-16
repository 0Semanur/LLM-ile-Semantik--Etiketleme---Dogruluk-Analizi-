DüğünBuketi AI Sohbet Analiz Sistemi
Bu proje, DüğünBuketi platformundan alınan müşteri konuşmalarını yapay zeka ile analiz ederek yanıtlanmamış soruları tespit eder, duygu analizi yapar ve müşteri hizmetleri kalitesini ölçer.
Özellikler
LLM Entegrasyonu
- OpenAI GPT-4o - En yeni model
- OpenAI GPT-4 ve GPT-3.5-turbo
- Groq (Llama3, Mixtral) - Ücretsiz
- Hugging Face - Ücretsiz
- Anthropic Claude
Analiz Çıktıları
- Bot Yanıt Verdi Mi? → Evet / Hayır (Müşteri hizmetleri kalitesi)
- Sentiment → Pozitif / Negatif / Nötr
- Konu Kategorisi → Düğün mekanı, Gelinlik, Fotoğrafçı, Müzik/DJ, vb.
- Detaylı İstatistikler → Token kullanımı, API çağrısı sayısı
Manuel Etiketleme Arayüzü
- Streamlit tabanlı web arayüzü
- LLM tahminlerini manuel doğrulama
- Gerçek zamanlı doğruluk hesaplama
- Görsel performans raporları
Doğruluk Analizi
- Confusion Matrix görselleştirme
- Precision, Recall, F1-Score metrikleri
- Hedef başarı oranı kontrolü (%85 hedef)
- Detaylı performans raporları
Kurulum
1. Gereksinimler
pip install -r requirements.txt
2. API Anahtarları
.env dosyası oluşturun:
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_key
GROQ_API_KEY=your_groq_key
HUGGINGFACE_API_KEY=your_hf_key
DEFAULT_LLM_PROVIDER=openai
DEFAULT_MODEL=gpt-4o
MAX_RETRIES=3
API_TIMEOUT=30
3. NLTK Verilerini İndirin
import nltk
nltk.download('punkt')
nltk.download('vader_lexicon')
Kullanım
Ana Workflow:
python main_workflow.py
Adım 1: LLM Analizi
1. JSON chat dosyasını seçin
2. LLM provider/model seçin (GPT-4o önerilen)
3. Otomatik analiz başlatın
Adım 2: Manuel Etiketleme
streamlit run enhanced_manual_labeling.py
- Web arayüzünde LLM tahminlerini doğrulayın
- Manuel etiketli dosyayı indirin
Adım 3: Doğruluk Analizi
- Manuel etiketli dosyayı yükleyin
- Detaylı performans raporu alın
Hızlı Test:
python test_data_generator.py
python main.py
Çıktı Formatları
LLM Analiz Sonuçları (CSV) ve Doğruluk Metrikleri (JSON) örnekleri içerir.
Düğün Buketi Kategorileri
Konu Kategorileri
- Düğün mekanı
- Gelinlik
- Fotoğrafçı
- Müzik/DJ
- Çiçek/Dekorasyon
- Davetiye
- Pasta/Catering
- Takı/Aksesuar
- Nikah şekeri
- Genel bilgi
- Fiyat sorgusu
- Rezervasyon
Sentiment Kategorileri
- Pozitif
- Nötr
- Negatif
Bot Yanıt Durumu
- Evet
- Hayır
