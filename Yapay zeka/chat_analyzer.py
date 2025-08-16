import pandas as pd
import json
import re
import sqlite3
from datetime import datetime
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import pickle
import os

class DugumBuketiChatAnalyzer:
    def __init__(self):
        """DüğünBuketi sohbet analiz sistemi"""
        self.categories = [
            'Düğün mekanı', 'Gelinlik', 'Fotoğrafçı', 'Müzik/DJ', 
            'Çiçek/Dekorasyon', 'Davetiye', 'Pasta/Catering', 
            'Video çekimi', 'Nikah şekeri', 'Takı/Aksesuar',
            'Genel bilgi', 'Fiyat sorgusu', 'Rezervasyon', 'Diğer'
        ]
        
        self.intents = [
            'Mekan arıyor', 'Ürün arıyor', 'Bilgi soruyor', 
            'Fiyat soruyor', 'Rezervasyon yapıyor', 'Şikayet ediyor',
            'Teşekkür ediyor', 'İptal ediyor', 'Değişiklik istiyor', 'Diğer'
        ]
        
        # Anahtar kelime sözlükleri
        self.category_keywords = {
            'Düğün mekanı': ['mekan', 'salon', 'bahçe', 'düğün salonu', 'organizasyon', 'yer'],
            'Gelinlik': ['gelinlik', 'elbise', 'gelin', 'kıyafet', 'dress'],
            'Fotoğrafçı': ['fotoğraf', 'çekim', 'albüm', 'kameraman', 'foto'],
            'Müzik/DJ': ['müzik', 'dj', 'ses sistemi', 'orkestra', 'canlı müzik'],
            'Çiçek/Dekorasyon': ['çiçek', 'dekorasyon', 'süsleme', 'aranjman', 'masa süsü'],
            'Davetiye': ['davetiye', 'kart', 'invitation', 'basım'],
            'Pasta/Catering': ['pasta', 'yemek', 'catering', 'ikram', 'menü'],
            'Video çekimi': ['video', 'klip', 'çekim', 'kameraman', 'montaj'],
            'Nikah şekeri': ['nikah şekeri', 'şeker', 'hediye', 'bonbon'],
            'Takı/Aksesuar': ['takı', 'aksesuar', 'yüzük', 'kolye', 'küpe'],
            'Fiyat sorgusu': ['fiyat', 'ücret', 'maliyet', 'para', 'bütçe', 'kaç lira'],
            'Rezervasyon': ['rezervasyon', 'randevu', 'tarih', 'müsait', 'booking']
        }
        
        self.intent_keywords = {
            'Mekan arıyor': ['mekan arıyorum', 'salon önerisi', 'yer önerisi', 'nerede'],
            'Ürün arıyor': ['arıyorum', 'istiyorum', 'lazım', 'gerek'],
            'Bilgi soruyor': ['nasıl', 'ne zaman', 'hangi', 'bilgi', 'soru'],
            'Fiyat soruyor': ['ne kadar', 'fiyat', 'ücret', 'maliyet'],
            'Rezervasyon yapıyor': ['rezervasyon', 'randevu', 'ayırt'],
            'Şikayet ediyor': ['şikayet', 'memnun değil', 'problem', 'sorun'],
            'Teşekkür ediyor': ['teşekkür', 'sağol', 'çok güzel', 'memnun'],
            'İptal ediyor': ['iptal', 'vazgeç', 'istemiyorum'],
            'Değişiklik istiyor': ['değiştir', 'farklı', 'başka']
        }
        
        # Duygu analizi için Türkçe pozitif/negatif kelimeler
        self.positive_words = [
            'güzel', 'harika', 'mükemmel', 'beğendim', 'teşekkür', 'memnun',
            'başarılı', 'kaliteli', 'profesyonel', 'tavsiye', 'süper'
        ]
        
        self.negative_words = [
            'kötü', 'berbat', 'memnun değil', 'şikayet', 'problem', 'sorun',
            'pahalı', 'kalitesiz', 'geç', 'yavaş', 'eksik'
        ]
        
    def preprocess_text(self, text):
        """Metni temizle ve normalize et"""
        if not isinstance(text, str):
            return ""
        
        # Küçük harfe çevir
        text = text.lower()
        
        # Türkçe karakterleri koru
        text = re.sub(r'[^\w\sçğıöşüÇĞIİÖŞÜ]', ' ', text)
        
        # Fazla boşlukları temizle
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def analyze_sentiment(self, text):
        """Duygu analizi yap"""
        text = self.preprocess_text(text)
        
        # Anahtar kelime bazlı analiz
        positive_count = sum(1 for word in self.positive_words if word in text)
        negative_count = sum(1 for word in self.negative_words if word in text)
        
        if positive_count > negative_count:
            return 'Pozitif'
        elif negative_count > positive_count:
            return 'Negatif'
        else:
            # TextBlob ile ek analiz
            try:
                blob = TextBlob(text)
                polarity = blob.sentiment.polarity
                if polarity > 0.1:
                    return 'Pozitif'
                elif polarity < -0.1:
                    return 'Negatif'
                else:
                    return 'Nötr'
            except:
                return 'Nötr'
    
    def classify_category(self, text):
        """Kategori sınıflandırması"""
        text = self.preprocess_text(text)
        
        category_scores = {}
        for category, keywords in self.category_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                category_scores[category] = score
        
        if category_scores:
            return max(category_scores, key=category_scores.get)
        else:
            return 'Diğer'
    
    def classify_intent(self, text):
        """Amaç (intent) sınıflandırması"""
        text = self.preprocess_text(text)
        
        intent_scores = {}
        for intent, keywords in self.intent_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                intent_scores[intent] = score
        
        if intent_scores:
            return max(intent_scores, key=intent_scores.get)
        else:
            return 'Diğer'
    
    def is_question_answered(self, conversation_history, current_message_index):
        """Sorunun yanıtlanıp yanıtlanmadığını kontrol et"""
        current_msg = conversation_history[current_message_index]
        
        # Soru işaretleri ve soru kelimeleri kontrol et
        question_indicators = ['?', 'nasıl', 'ne zaman', 'nerede', 'hangi', 'kaç', 'kim']
        
        text = self.preprocess_text(current_msg.get('message', ''))
        is_question = any(indicator in text for indicator in question_indicators)
        
        if not is_question:
            return 'Hayır'  # Soru değilse yanıtlanma durumu önemli değil
        
        # Sonraki mesajlarda yanıt arayalım
        for i in range(current_message_index + 1, min(current_message_index + 3, len(conversation_history))):
            next_msg = conversation_history[i]
            
            # Farklı kişiden gelen mesaj mı kontrol et
            if (next_msg.get('sender') != current_msg.get('sender') or 
                next_msg.get('user_type') != current_msg.get('user_type')):
                
                next_text = self.preprocess_text(next_msg.get('message', ''))
                
                # Yanıt belirten kelimeler
                answer_indicators = [
                    'evet', 'hayır', 'tabii', 'elbette', 'maalesef', 
                    'şöyle', 'şu şekilde', 'bilgi', 'cevap'
                ]
                
                if any(indicator in next_text for indicator in answer_indicators):
                    return 'Evet'
                
                # Mesaj uzunluğu kontrolü (detaylı yanıt)
                if len(next_text.split()) > 5:
                    return 'Evet'
        
        return 'Hayır'
    
    def analyze_conversation(self, json_data):
        """JSON formatındaki konuşmayı analiz et"""
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
        
        for i, message in enumerate(messages):
            message_text = message.get('message', '')
            
            if not message_text.strip():
                continue
            
            analysis = {
                'message_id': message.get('id', i),
                'timestamp': message.get('timestamp', datetime.now().isoformat()),
                'sender': message.get('sender', 'unknown'),
                'message': message_text,
                'yanıtlanmış_mı': self.is_question_answered(messages, i),
                'sentiment': self.analyze_sentiment(message_text),
                'kategori': self.classify_category(message_text),
                'intent': self.classify_intent(message_text)
            }
            
            results.append(analysis)
        
        return results
    
    def save_to_csv(self, results, filename='dugum_buketi_analiz.csv'):
        """Sonuçları CSV dosyasına kaydet"""
        df = pd.DataFrame(results)
        filepath = os.path.join(os.getcwd(), filename)
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"Sonuçlar CSV dosyasına kaydedildi: {filepath}")
        return filepath
    
    def save_to_sqlite(self, results, db_name='dugum_buketi_analiz.db'):
        """Sonuçları SQLite veritabanına kaydet"""
        filepath = os.path.join(os.getcwd(), db_name)
        conn = sqlite3.connect(filepath)
        
        df = pd.DataFrame(results)
        df.to_sql('chat_analysis', conn, if_exists='replace', index=False)
        
        conn.close()
        print(f"Sonuçlar SQLite veritabanına kaydedildi: {filepath}")
        return filepath
    
    def generate_report(self, results):
        """Analiz raporu oluştur"""
        df = pd.DataFrame(results)
        
        report = {
            'toplam_mesaj': len(df),
            'yanıtlanmamış_soru': len(df[df['yanıtlanmış_mı'] == 'Hayır']),
            'sentiment_dağılımı': df['sentiment'].value_counts().to_dict(),
            'kategori_dağılımı': df['kategori'].value_counts().to_dict(),
            'intent_dağılımı': df['intent'].value_counts().to_dict()
        }
        
        return report