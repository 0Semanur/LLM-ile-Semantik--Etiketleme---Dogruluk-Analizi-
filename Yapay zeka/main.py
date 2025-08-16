from chat_analyzer import DugumBuketiChatAnalyzer
import json
import pandas as pd

def main():
    # Analiz sistemi oluştur
    analyzer = DugumBuketiChatAnalyzer()
    
    # Örnek JSON verisi (gerçek veri yerine)
    sample_data = {
        "messages": [
            {
                "id": 1,
                "timestamp": "2024-01-15T10:30:00",
                "sender": "müşteri_1",
                "user_type": "customer",
                "message": "Merhaba, düğün mekanı arıyorum. Bahçeli bir yer önerebilir misiniz?"
            },
            {
                "id": 2,
                "timestamp": "2024-01-15T10:32:00",
                "sender": "destek_1",
                "user_type": "support",
                "message": "Merhaba! Tabii ki. Hangi bölgede düğün mekanı arıyorsunuz?"
            },
            {
                "id": 3,
                "timestamp": "2024-01-15T10:35:00",
                "sender": "müşteri_1",
                "user_type": "customer",
                "message": "İstanbul Avrupa yakasında olsun. Fiyatları nasıl?"
            },
            {
                "id": 4,
                "timestamp": "2024-01-15T10:40:00",
                "sender": "müşteri_2",
                "user_type": "customer",
                "message": "Gelinlik modelleri çok güzel! Teşekkür ederim."
            },
            {
                "id": 5,
                "timestamp": "2024-01-15T11:00:00",
                "sender": "müşteri_3",
                "user_type": "customer",
                "message": "Fotoğrafçı rezervasyonu nasıl yapabilirim? Acil cevap bekliyorum."
            }
        ]
    }
    
    print("DüğünBuketi Sohbet Analizi Başlatılıyor...")
    print("=" * 50)
    
    # Analiz yap
    results = analyzer.analyze_conversation(sample_data)
    
    # Sonuçları göster
    print("\nAnaliz Sonuçları:")
    print("-" * 30)
    
    for result in results:
        print(f"\nMesaj ID: {result['message_id']}")
        print(f"Gönderen: {result['sender']}")
        print(f"Mesaj: {result['message'][:100]}...")
        print(f"Yanıtlanmış mı: {result['yanıtlanmış_mı']}")
        print(f"Duygu: {result['sentiment']}")
        print(f"Kategori: {result['kategori']}")
        print(f"Amaç: {result['intent']}")
        print("-" * 30)
    
    # CSV ve SQLite'a kaydet
    csv_file = analyzer.save_to_csv(results)
    db_file = analyzer.save_to_sqlite(results)
    
    # Rapor oluştur
    report = analyzer.generate_report(results)
    print(f"\n📊 ÖZET RAPOR:")
    print(f"Toplam Mesaj: {report['toplam_mesaj']}")
    print(f"Yanıtlanmamış Soru: {report['yanıtlanmamış_soru']}")
    print(f"\nDuygu Dağılımı: {report['sentiment_dağılımı']}")
    print(f"Kategori Dağılımı: {report['kategori_dağılımı']}")
    print(f"Amaç Dağılımı: {report['intent_dağılımı']}")
    
    return results, csv_file, db_file

def analyze_custom_json(json_file_path):
    """Kendi JSON dosyanızı analiz etmek için"""
    analyzer = DugumBuketiChatAnalyzer()
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        results = analyzer.analyze_conversation(data)
        
        # Dosya adından çıktı adları oluştur
        base_name = json_file_path.replace('.json', '')
        csv_file = analyzer.save_to_csv(results, f"{base_name}_analiz.csv")
        db_file = analyzer.save_to_sqlite(results, f"{base_name}_analiz.db")
        
        report = analyzer.generate_report(results)
        print("Analiz tamamlandı!")
        print(f"CSV: {csv_file}")
        print(f"SQLite: {db_file}")
        
        return results, report
        
    except Exception as e:
        print(f"Hata: {e}")
        return None, None

if __name__ == "__main__":
    # Örnek analiz çalıştır
    results, csv_file, db_file = main()
    
    print(f"\n✅ Analiz tamamlandı!")
    print(f"📁 CSV dosyası: {csv_file}")
    print(f"🗄️ SQLite dosyası: {db_file}")
    
    # Kendi JSON dosyanızı analiz etmek için:
    # results, report = analyze_custom_json("your_chat_data.json")