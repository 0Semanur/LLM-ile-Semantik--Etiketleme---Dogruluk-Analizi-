import json
import random
from datetime import datetime, timedelta

def generate_sample_data():
    """Test için örnek sohbet verisi oluştur"""
    
    sample_messages = [
        # Düğün mekanı soruları
        "Merhaba, düğün mekanı arıyorum. Bahçeli bir yer var mı?",
        "İstanbul'da 200 kişilik düğün salonu önerebilir misiniz?",
        "Düğün mekanı fiyatları nasıl? Bütçem 50.000 TL",
        
        # Gelinlik soruları  
        "Gelinlik modelleri görebilir miyim?",
        "Prenses model gelinlik var mı? Fiyatı ne kadar?",
        "Gelinlik ölçü aldırma nasıl oluyor?",
        
        # Fotoğrafçı soruları
        "Düğün fotoğrafçısı rezervasyonu nasıl yapabilirim?",
        "Fotoğraf paketleriniz neler? Fiyat listesi var mı?",
        "Düğün albümü kaç günde hazır oluyor?",
        
        # Pozitif yorumlar
        "Çok güzel hizmet veriyorsunuz, teşekkür ederim!",
        "Fotoğraflar harika olmuş, çok memnun kaldık",
        "Personel çok ilgili ve profesyonel",
        
        # Şikayetler
        "Randevuma geç kaldınız, memnun değilim",
        "Fiyatlar çok pahalı, başka seçenek var mı?",
        "Aradığım ürünü bulamadım, yardım edebilir misiniz?",
        
        # Genel sorular
        "Çalışma saatleriniz nedir?",
        "Hangi şehirlerde hizmet veriyorsunuz?",
        "Online ödeme yapabilir miyim?",
        
        # Yanıtlar
        "Tabii ki! Size uygun seçenekleri gösterebilirim",
        "Elbette, detaylı bilgi için randevu alabilirsiniz",
        "Maalesef o tarih dolu, başka tarih önerebilirim",
        "Teşekkür ederiz! Memnuniyetiniz bizim için önemli",
        "Özür dileriz, sorununuzu çözmek için elimizden geleni yapacağız"
    ]
    
    senders = ['müşteri_1', 'müşteri_2', 'müşteri_3', 'destek_1', 'destek_2']
    user_types = ['customer', 'support']
    
    messages = []
    base_time = datetime.now() - timedelta(days=7)
    
    for i in range(50):  # 50 mesaj oluştur
        sender = random.choice(senders)
        user_type = 'customer' if 'müşteri' in sender else 'support'
        
        message = {
            "id": i + 1,
            "timestamp": (base_time + timedelta(minutes=i*10)).isoformat(),
            "sender": sender,
            "user_type": user_type,
            "message": random.choice(sample_messages)
        }
        messages.append(message)
    
    conversation_data = {
        "conversation_id": "dugum_buketi_001",
        "date": datetime.now().isoformat(),
        "messages": messages
    }
    
    # JSON dosyasına kaydet
    with open('sample_chat_data.json', 'w', encoding='utf-8') as f:
        json.dump(conversation_data, f, ensure_ascii=False, indent=2)
    
    print("Örnek veri oluşturuldu: sample_chat_data.json")
    return conversation_data

if __name__ == "__main__":
    generate_sample_data()