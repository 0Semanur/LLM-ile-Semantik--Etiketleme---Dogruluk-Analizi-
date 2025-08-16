import openai
import anthropic
import json
import pandas as pd
import os
from datetime import datetime
from typing import Dict, List, Tuple
import time
from dotenv import load_dotenv

# .env dosyasından API anahtarlarını yükle
load_dotenv()

class LLMChatAnalyzer:
    def __init__(self, provider="openai"):
        """
        LLM tabanlı sohbet analiz sistemi
        
        Args:
            provider (str): "openai" veya "anthropic"
        """
        self.provider = provider
        self.setup_client()
        
        # Prompt şablonları
        self.sentiment_prompt = """
        Aşağıdaki Türkçe mesajın duygusal tonunu analiz et ve sadece şu seçeneklerden birini döndür:
        - Pozitif
        - Negatif  
        - Nötr
        
        Mesaj: "{text}"
        
        Cevap (sadece tek kelime):
        """
        
        self.topic_prompt = """
        Aşağıdaki düğün sektörü ile ilgili Türkçe mesajın konusunu belirle ve sadece şu kategorilerden birini döndür:
        - Düğün mekanı
        - Gelinlik
        - Fotoğrafçı
        - Müzik/DJ
        - Çiçek/Dekorasyon
        - Davetiye
        - Pasta/Catering
        - Video çekimi
        - Nikah şekeri
        - Takı/Aksesuar
        - Genel bilgi
        - Fiyat sorgusu
        - Rezervasyon
        - Diğer
        
        Mesaj: "{text}"
        
        Cevap (sadece kategori adı):
        """
        
        self.bot_response_prompt = """
        Aşağıdaki konuşma geçmişinde, müşteri sorusu bot/destek ekibi tarafından yanıtlanmış mı?
        
        Konuşma:
        {conversation}
        
        Son müşteri mesajı: "{customer_message}"
        
        Bu soruya bot veya destek ekibi yanıt vermiş mi? Sadece "Evet" veya "Hayır" ile cevapla:
        """
    
    def setup_client(self):
        """API istemcisini ayarla"""
        if self.provider == "openai":
            self.client = openai.OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )
            self.model = "gpt-3.5-turbo"
        elif self.provider == "anthropic":
            self.client = anthropic.Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
            self.model = "claude-3-haiku-20240307"
        else:
            raise ValueError("Desteklenen provider'lar: 'openai' veya 'anthropic'")
    
    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """LLM API çağrısı yap"""
        for attempt in range(max_retries):
            try:
                if self.provider == "openai":
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "Sen Türkçe metin analizi yapan bir asistansın. Kısa ve net cevaplar ver."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=50,
                        temperature=0.1
                    )
                    return response.choices[0].message.content.strip()
                
                elif self.provider == "anthropic":
                    response = self.client.messages.create(
                        model=self.model,
                        max_tokens=50,
                        temperature=0.1,
                        messages=[
                            {"role": "user", "content": prompt}
                        ]
                    )
                    return response.content[0].text.strip()
                    
            except Exception as e:
                print(f"API çağrısı hatası (deneme {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return "Hata"
        
        return "Hata"
    
    def analyze_sentiment_llm(self, text: str) -> str:
        """LLM ile duygu analizi"""
        prompt = self.sentiment_prompt.format(text=text)
        result = self.call_llm(prompt)
        
        # Sonucu temizle ve doğrula
        result = result.strip().title()
        if result in ["Pozitif", "Negatif", "Nötr"]:
            return result
        else:
            return "Nötr"  # Varsayılan değer
    
    def analyze_topic_llm(self, text: str) -> str:
        """LLM ile konu analizi"""
        prompt = self.topic_prompt.format(text=text)
        result = self.call_llm(prompt)
        
        # Geçerli kategoriler listesi
        valid_categories = [
            "Düğün mekanı", "Gelinlik", "Fotoğrafçı", "Müzik/DJ", 
            "Çiçek/Dekorasyon", "Davetiye", "Pasta/Catering", 
            "Video çekimi", "Nikah şekeri", "Takı/Aksesuar",
            "Genel bilgi", "Fiyat sorgusu", "Rezervasyon", "Diğer"
        ]
        
        # En yakın kategoriyi bul
        result = result.strip()
        for category in valid_categories:
            if category.lower() in result.lower():
                return category
        
        return "Diğer"  # Varsayılan değer
    
    def analyze_bot_response_llm(self, conversation_history: List[Dict], current_message_index: int) -> str:
        """LLM ile bot yanıt analizi"""
        current_msg = conversation_history[current_message_index]
        
        # Konuşma geçmişini hazırla (son 5 mesaj)
        start_idx = max(0, current_message_index - 2)
        end_idx = min(len(conversation_history), current_message_index + 3)
        
        conversation_text = ""
        for i in range(start_idx, end_idx):
            msg = conversation_history[i]
            sender_type = "Müşteri" if msg.get('user_type') == 'customer' else "Destek"
            conversation_text += f"{sender_type}: {msg.get('message', '')}\n"
        
        prompt = self.bot_response_prompt.format(
            conversation=conversation_text,
            customer_message=current_msg.get('message', '')
        )
        
        result = self.call_llm(prompt)
        
        # Sonucu temizle ve doğrula
        result = result.strip().title()
        if "Evet" in result:
            return "Evet"
        elif "Hayır" in result:
            return "Hayır"
        else:
            return "Hayır"  # Varsayılan değer
    
    def analyze_conversation_llm(self, json_data) -> List[Dict]:
        """Konuşmayı LLM ile analiz et"""
        results = []
        
        if isinstance(json_data, str):
            conversation = json.loads(json_data)
        else:
            conversation = json_data
        
        # Konuşma geçmişi listesi olarak al
        if isinstance(conversation, dict) and 'messages' in conversation:
            messages = conversation['messages']
        elif isinstance(conversation, list):
            messages = conversation
        else:
            messages = [conversation]
        
        print(f"LLM ile {len(messages)} mesaj analiz ediliyor...")
        
        for i, message in enumerate(messages):
            message_text = message.get('message', '')
            
            if not message_text.strip():
                continue
            
            print(f"Mesaj {i+1}/{len(messages)} analiz ediliyor...")
            
            # LLM analizleri
            sentiment = self.analyze_sentiment_llm(message_text)
            topic = self.analyze_topic_llm(message_text)
            bot_response = self.analyze_bot_response_llm(messages, i)
            
            analysis = {
                'message_id': message.get('id', i),
                'timestamp': message.get('timestamp', datetime.now().isoformat()),
                'sender': message.get('sender', 'unknown'),
                'message': message_text,
                'llm_sentiment': sentiment,
                'llm_topic': topic,
                'llm_bot_response': bot_response
            }
            
            results.append(analysis)
            
            # API rate limiting için kısa bekleme
            time.sleep(0.5)
        
        print("LLM analizi tamamlandı!")
        return results
    
    def save_llm_results(self, results: List[Dict], filename: str = None) -> str:
        """LLM sonuçlarını kaydet"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"llm_analiz_{timestamp}.csv"
        
        df = pd.DataFrame(results)
        filepath = os.path.join(os.getcwd(), filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"LLM analiz sonuçları kaydedildi: {filepath}")
        return filepath