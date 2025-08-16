import openai
import json
import pandas as pd
import os
import time
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv
import logging
import requests

# Logging ayarla
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# .env dosyasından API anahtarlarını yükle
load_dotenv()

class EnhancedLLMAnalyzer:
    def __init__(self, provider="openai", model=None):
        """
        Gelişmiş LLM tabanlı sohbet analiz sistemi
        
        Args:
            provider (str): "openai", "anthropic", "groq", "huggingface"
            model (str): Kullanılacak model adı
        """
        self.provider = provider
        
        # Provider'a göre varsayılan model seç
        if model is None:
            if provider == "groq":
                self.model = "llama3-8b-8192"
            elif provider == "openai":
                self.model = "gpt-4o"  # En yeni GPT-4o modeli
            elif provider == "huggingface":
                self.model = "microsoft/DialoGPT-medium"
            else:
                self.model = "gpt-4o"
        else:
            self.model = model
            
        self.setup_client()
        
        # DüğünBuketi.com kategorileri (web araştırmasından)
        self.dugum_buketi_categories = [
            "Düğün Mekanı",
            "Düğün Organizasyonu", 
            "Gelinlik",
            "Fotoğrafçı",
            "Video Çekimi",
            "Müzik/DJ",
            "Çiçek/Dekorasyon",
            "Davetiye",
            "Pasta/Catering",
            "Nikah Şekeri",
            "Takı/Aksesuar",
            "Düğün Arabası",
            "Düğün Dansı",
            "Genel Bilgi",
            "Fiyat Sorgusu",
            "Rezervasyon",
            "Şikayet",
            "Diğer"
        ]
        
        # Gelişmiş prompt şablonları
        self.setup_prompts()
        
        # API çağrı sayacı ve maliyet takibi
        self.api_calls = 0
        self.total_tokens = 0
        
    def setup_client(self):
        """API istemcisini ayarla"""
        if self.provider == "openai":
            try:
                import openai
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key or api_key == "your_openai_api_key_here":
                    raise ValueError("""
🔑 OPENAI_API_KEY bulunamadı!

API anahtarını almak için:
1. https://platform.openai.com/api-keys adresine gidin
2. Hesap oluşturun veya giriş yapın
3. 'Create new secret key' butonuna tıklayın
4. Anahtarı kopyalayın
5. .env dosyasında OPENAI_API_KEY=your_key_here satırını güncelleyin

Mevcut modeller:
- gpt-4o (en yeni, önerilen)
- gpt-4-turbo
- gpt-4
- gpt-3.5-turbo
                    """)
                
                self.client = openai.OpenAI(api_key=api_key)
                logger.info(f"✅ OpenAI client başlatıldı. Model: {self.model}")
                
                # Model doğrulaması
                if self.model not in ["gpt-4o", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"]:
                    logger.warning(f"⚠️ Model '{self.model}' doğrulanamadı. gpt-4o kullanılacak.")
                    self.model = "gpt-4o"
                    
            except ImportError:
                raise ValueError("OpenAI kütüphanesi yüklü değil! Çalıştırın: pip install openai")
                
        elif self.provider == "groq":
            try:
                from groq import Groq
                api_key = os.getenv("GROQ_API_KEY")
                if not api_key:
                    raise ValueError("GROQ_API_KEY environment variable bulunamadı!")
                self.client = Groq(api_key=api_key)
                logger.info(f"Groq client başlatıldı. Model: {self.model}")
            except ImportError:
                raise ValueError("Groq kütüphanesi yüklü değil! pip install groq")
                
        elif self.provider == "huggingface":
            try:
                from huggingface_hub import InferenceClient
                api_key = os.getenv("HUGGINGFACE_API_KEY")
                if not api_key:
                    raise ValueError("HUGGINGFACE_API_KEY environment variable bulunamadı!")
                self.client = InferenceClient(token=api_key)
                logger.info(f"Hugging Face client başlatıldı. Model: {self.model}")
            except ImportError:
                raise ValueError("Hugging Face kütüphanesi yüklü değil! pip install huggingface_hub")
        else:
            raise ValueError(f"Desteklenmeyen provider: {self.provider}")
    
    def setup_prompts(self):
        """Gelişmiş prompt şablonlarını ayarla"""
        
        # Sistem mesajı - GPT-4o için optimize edilmiş
        self.system_message = """Sen düğün sektöründe uzman bir AI asistanısın. Türkçe müşteri mesajlarını analiz ediyorsun.

Görevin: Her mesaj için sentiment, konu ve bot yanıt durumunu doğru şekilde belirlemek.

Önemli kurallar:
- Sadece belirtilen kategorilerden seçim yap
- Kısa, net ve tutarlı cevaplar ver
- Türkçe dil kurallarına uy
- Belirsiz durumlarda en yakın kategoriyi seç"""

        # Sentiment analizi prompt'u - GPT-4o için optimize edilmiş
        self.sentiment_prompt = """
Aşağıdaki müşteri mesajının duygusal tonunu analiz et.

MESAJ: "{text}"

KURALLAR:
- Sadece şu 3 seçenekten birini döndür: Pozitif, Negatif, Nötr
- Pozitif: Memnuniyet, teşekkür, beğeni, heyecan, övgü ifadeleri
- Negatif: Şikayet, memnuniyetsizlik, kızgınlık, hayal kırıklığı, eleştiri
- Nötr: Soru sorma, bilgi isteme, tarafsız ifadeler, normal konuşma

ÖRNEKLER:
"Çok güzel hizmet, teşekkürler!" → Pozitif
"Harika bir deneyimdi, kesinlikle tavsiye ederim" → Pozitif
"Geç kaldınız, memnun değilim" → Negatif
"Bu kadar pahalı olması normal değil" → Negatif
"Düğün mekanı arıyorum" → Nötr
"Fiyatları öğrenebilir miyim?" → Nötr

CEVAP (sadece tek kelime):"""

        # Konu sınıflandırması prompt'u - GPT-4o için optimize edilmiş
        categories_text = ", ".join(self.dugum_buketi_categories)
        
        self.topic_prompt = """
Aşağıdaki düğün sektörü mesajının ana konusunu belirle.

MESAJ: "{text}"

KATEGORİLER: """ + categories_text + """

KURALLAR:
- Sadece yukarıdaki kategorilerden birini seç
- Mesajın ana konusuna ve amacına odaklan
- Belirsizse en yakın kategoriyi seç
- Fiyat soruları için "Fiyat Sorgusu" kullan
- Şikayet ifadeleri için "Şikayet" kullan

ÖRNEKLER:
"Gelinlik denemek istiyorum" → Gelinlik
"Düğün fotoğrafçısı arıyorum" → Fotoğrafçı
"Fiyatlar nasıl?" → Fiyat Sorgusu
"Ne kadar tutuyor?" → Fiyat Sorgusu
"Memnun değilim, şikayet ediyorum" → Şikayet
"Randevu almak istiyorum" → Rezervasyon
"Düğün mekanı önerisi var mı?" → Düğün Mekanı

CEVAP (sadece kategori adı):"""

        # Bot yanıt analizi prompt'u - GPT-4o için optimize edilmiş
        self.bot_response_prompt = """
Aşağıdaki konuşmada müşteri sorusu/talebi destek ekibi tarafından yanıtlanmış mı?

KONUŞMA GEÇMİŞİ:
{conversation_context}

SON MÜŞTERİ MESAJI: "{customer_message}"

KURALLAR:
- Müşteri mesajından sonra destek/bot yanıtı var mı kontrol et
- Yanıt müşterinin sorusunu/talebini karşılıyor mu?
- Sadece "Evet" veya "Hayır" döndür
- Otomatik yanıtlar da "Evet" sayılır

ÖRNEKLER:
Müşteri: "Fiyat ne kadar?"
Destek: "150 TL'den başlıyor" → Evet

Müşteri: "Randevu alabilir miyim?"
Destek: "Tabii ki, hangi tarih uygun?" → Evet

Müşteri: "Teşekkür ederim"
(Yanıt yok veya sadece müşteri mesajları devam ediyor) → Hayır

CEVAP (Evet/Hayır):"""

    def call_llm_with_retry(self, messages: List[Dict], max_retries: int = 3) -> str:
        """LLM API çağrısı yap (retry mekanizması ile)"""
        
        for attempt in range(max_retries):
            try:
                if self.provider == "groq":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=50,
                        temperature=0.1,
                        top_p=0.9
                    )
                    
                elif self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=50,
                        temperature=0.1,
                        top_p=0.9,
                        frequency_penalty=0,
                        presence_penalty=0
                    )
                    
                elif self.provider == "huggingface":
                    # Hugging Face için farklı format
                    prompt = f"{messages[0]['content']}\n\n{messages[1]['content']}"
                    response = self.client.text_generation(
                        prompt=prompt,
                        model=self.model,
                        max_new_tokens=50,
                        temperature=0.1
                    )
                    result = response.strip()
                    self.api_calls += 1
                    logger.info(f"LLM Response: {result}")
                    return result
                
                # İstatistikleri güncelle
                self.api_calls += 1
                if hasattr(response, 'usage') and response.usage:
                    self.total_tokens += response.usage.total_tokens
                
                result = response.choices[0].message.content.strip()
                logger.info(f"LLM Response: {result}")
                return result
                
            except Exception as e:
                logger.warning(f"API çağrısı hatası (deneme {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"API çağrısı başarısız: {e}")
                    return "Hata"
        
        return "Hata"

    def analyze_sentiment_enhanced(self, text: str) -> str:
        """Gelişmiş sentiment analizi"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.sentiment_prompt.format(text=text)}
        ]
        
        result = self.call_llm_with_retry(messages)
        
        # Sonucu temizle ve doğrula
        result = result.strip().title()
        valid_sentiments = ["Pozitif", "Negatif", "Nötr"]
        
        if result in valid_sentiments:
            return result
        else:
            # Fallback: Anahtar kelime analizi
            return self._fallback_sentiment_analysis(text)
    
    def analyze_topic_enhanced(self, text: str) -> str:
        """Gelişmiş konu analizi"""
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.topic_prompt.format(text=text)}
        ]
        
        result = self.call_llm_with_retry(messages)
        
        # En yakın kategoriyi bul
        result = result.strip()
        for category in self.dugum_buketi_categories:
            if category.lower() in result.lower() or result.lower() in category.lower():
                return category
        
        # Fallback: Anahtar kelime analizi
        return self._fallback_topic_analysis(text)
    
    def analyze_bot_response_enhanced(self, conversation_history: List[Dict], current_index: int) -> str:
        """Gelişmiş bot yanıt analizi"""
        current_msg = conversation_history[current_index]
        
        # Konuşma bağlamını hazırla
        context_messages = []
        start_idx = max(0, current_index - 2)
        end_idx = min(len(conversation_history), current_index + 3)
        
        for i in range(start_idx, end_idx):
            msg = conversation_history[i]
            sender_type = "Müşteri" if msg.get('user_type') == 'customer' else "Destek"
            context_messages.append(f"{sender_type}: {msg.get('message', '')}")
        
        conversation_context = "\n".join(context_messages)
        
        messages = [
            {"role": "system", "content": self.system_message},
            {"role": "user", "content": self.bot_response_prompt.format(
                conversation_context=conversation_context,
                customer_message=current_msg.get('message', '')
            )}
        ]
        
        result = self.call_llm_with_retry(messages)
        
        # Sonucu temizle ve doğrula
        if "evet" in result.lower():
            return "Evet"
        elif "hayır" in result.lower():
            return "Hayır"
        else:
            # Fallback: Basit kural tabanlı analiz
            return self._fallback_bot_response_analysis(conversation_history, current_index)
    
    def _fallback_sentiment_analysis(self, text: str) -> str:
        """Fallback sentiment analizi"""
        text_lower = text.lower()
        
        positive_words = ['güzel', 'harika', 'mükemmel', 'teşekkür', 'memnun', 'beğendim', 'süper', 'muhteşem']
        negative_words = ['kötü', 'berbat', 'şikayet', 'memnun değil', 'problem', 'geç', 'pahalı', 'kızgın']
        
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)
        
        if pos_count > neg_count:
            return "Pozitif"
        elif neg_count > pos_count:
            return "Negatif"
        else:
            return "Nötr"
    
    def _fallback_topic_analysis(self, text: str) -> str:
        """Fallback konu analizi"""
        text_lower = text.lower()
        
        keyword_mapping = {
            "Düğün Mekanı": ["mekan", "salon", "bahçe", "düğün salonu", "yer"],
            "Gelinlik": ["gelinlik", "elbise", "gelin", "kıyafet"],
            "Fotoğrafçı": ["fotoğraf", "çekim", "albüm", "kameraman"],
            "Fiyat Sorgusu": ["fiyat", "ücret", "maliyet", "ne kadar", "para", "tutar"],
            "Rezervasyon": ["rezervasyon", "randevu", "tarih", "saat"],
            "Şikayet": ["şikayet", "memnun değil", "problem", "sorun"]
        }
        
        for category, keywords in keyword_mapping.items():
            if any(keyword in text_lower for keyword in keywords):
                return category
        
        return "Genel Bilgi"
    
    def _fallback_bot_response_analysis(self, conversation_history: List[Dict], current_index: int) -> str:
        """Fallback bot yanıt analizi"""
        current_msg = conversation_history[current_index]
        
        # Sonraki mesajları kontrol et
        for i in range(current_index + 1, min(len(conversation_history), current_index + 3)):
            next_msg = conversation_history[i]
            if next_msg.get('user_type') == 'support':
                return "Evet"
        
        return "Hayır"

    def analyze_conversation(self, conversation_data: List[Dict]) -> pd.DataFrame:
        """Tüm konuşmayı analiz et"""
        results = []
        
        logger.info(f"🔍 {len(conversation_data)} mesaj analiz ediliyor...")
        logger.info(f"🤖 Provider: {self.provider}")
        logger.info(f"🧠 Model: {self.model}")
        
        for i, message in enumerate(conversation_data):
            try:
                logger.info(f"📝 Mesaj {i+1}/{len(conversation_data)} analiz ediliyor...")
                
                # Temel bilgiler
                result = {
                    'message_id': message.get('message_id', i+1),
                    'timestamp': message.get('timestamp', ''),
                    'sender': message.get('sender', ''),
                    'user_type': message.get('user_type', ''),
                    'message': message.get('message', ''),
                }
                
                # LLM analizleri
                text = message.get('message', '')
                
                if text.strip():
                    # Sentiment analizi
                    sentiment = self.analyze_sentiment_enhanced(text)
                    result['llm_sentiment'] = sentiment
                    
                    # Konu analizi
                    topic = self.analyze_topic_enhanced(text)
                    result['llm_topic'] = topic
                    
                    # Bot yanıt analizi
                    bot_response = self.analyze_bot_response_enhanced(conversation_data, i)
                    result['llm_bot_response'] = bot_response
                    
                    logger.info(f"✅ Analiz tamamlandı: {sentiment} | {topic} | {bot_response}")
                else:
                    result['llm_sentiment'] = 'Nötr'
                    result['llm_topic'] = 'Genel Bilgi'
                    result['llm_bot_response'] = 'Hayır'
                
                # Metadata
                result['analysis_timestamp'] = datetime.now().isoformat()
                result['provider_used'] = self.provider
                result['model_used'] = self.model
                
                results.append(result)
                
                # Rate limiting
                time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"❌ Mesaj {i+1} analiz hatası: {e}")
                # Hata durumunda varsayılan değerler
                result = {
                    'message_id': message.get('message_id', i+1),
                    'timestamp': message.get('timestamp', ''),
                    'sender': message.get('sender', ''),
                    'user_type': message.get('user_type', ''),
                    'message': message.get('message', ''),
                    'llm_sentiment': 'Hata',
                    'llm_topic': 'Hata',
                    'llm_bot_response': 'Hata',
                    'analysis_timestamp': datetime.now().isoformat(),
                    'provider_used': self.provider,
                    'model_used': self.model
                }
                results.append(result)
        
        df = pd.DataFrame(results)
        
        # İstatistikleri yazdır
        logger.info(f"\n📊 ANALİZ İSTATİSTİKLERİ:")
        logger.info(f"✅ Toplam mesaj: {len(conversation_data)}")
        logger.info(f"🔄 API çağrısı: {self.api_calls}")
        logger.info(f"🎯 Token kullanımı: {self.total_tokens}")
        logger.info(f"🤖 Provider: {self.provider}")
        logger.info(f"🧠 Model: {self.model}")
        
        return df

    def save_analysis_results(self, df: pd.DataFrame, output_prefix: str = "enhanced_llm_analysis") -> Tuple[str, str]:
        """Analiz sonuçlarını kaydet"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # CSV dosyası
        csv_filename = f"{output_prefix}_{self.provider}_{timestamp}.csv"
        df.to_csv(csv_filename, index=False, encoding='utf-8-sig')
        
        # Metadata dosyası
        metadata = {
            'analysis_timestamp': datetime.now().isoformat(),
            'provider': self.provider,
            'model': self.model,
            'total_messages': len(df),
            'api_calls': self.api_calls,
            'total_tokens': self.total_tokens,
            'sentiment_distribution': df['llm_sentiment'].value_counts().to_dict(),
            'topic_distribution': df['llm_topic'].value_counts().to_dict(),
            'bot_response_distribution': df['llm_bot_response'].value_counts().to_dict()
        }
        
        metadata_filename = f"{output_prefix}_{self.provider}_{timestamp}_metadata.json"
        with open(metadata_filename, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Sonuçlar kaydedildi:")
        logger.info(f"📄 CSV: {csv_filename}")
        logger.info(f"📋 Metadata: {metadata_filename}")
        
        return csv_filename, metadata_filename

def main():
    """Test fonksiyonu"""
    # GPT-4o ile test
    analyzer = EnhancedLLMAnalyzer(provider="openai", model="gpt-4o")
    
    # Test verisi
    test_data = [
        {
            "message_id": 1,
            "timestamp": "2024-12-15T09:00:00",
            "sender": "müşteri_1",
            "user_type": "customer",
            "message": "Merhaba, düğün mekanı arıyorum. Bahçeli bir yer var mı?"
        }
    ]
    
    # Analiz yap
    results = analyzer.analyze_conversation(test_data)
    print(results)

if __name__ == "__main__":
    main()