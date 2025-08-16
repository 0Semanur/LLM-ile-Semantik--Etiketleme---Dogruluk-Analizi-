import json
import pandas as pd
from datetime import datetime
import os

class ChatDataConverter:
    def __init__(self):
        """Chat verilerini JSON formatına çeviren sınıf"""
        pass
    
    def excel_to_json(self, excel_file_path: str, output_file: str = None):
        """Excel dosyasından JSON'a çevir"""
        try:
            # Excel dosyasını oku
            df = pd.read_excel(excel_file_path)
            
            # Gerekli sütunları kontrol et
            required_columns = ['message', 'sender', 'timestamp']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Eksik sütunlar: {missing_columns}")
                print("📋 Gerekli sütunlar: message, sender, timestamp")
                print("📋 İsteğe bağlı: user_type, id")
                return None
            
            # JSON formatına çevir
            messages = []
            for i, row in df.iterrows():
                message = {
                    "id": row.get('id', i + 1),
                    "timestamp": str(row['timestamp']),
                    "sender": str(row['sender']),
                    "user_type": row.get('user_type', 'customer'),
                    "message": str(row['message'])
                }
                messages.append(message)
            
            conversation_data = {
                "conversation_id": f"chat_data_{datetime.now().strftime('%Y%m%d')}",
                "date": datetime.now().isoformat(),
                "total_messages": len(messages),
                "messages": messages
            }
            
            # JSON dosyasına kaydet
            if output_file is None:
                output_file = excel_file_path.replace('.xlsx', '.json').replace('.xls', '.json')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ Excel verisi JSON'a çevrildi: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Hata: {e}")
            return None
    
    def csv_to_json(self, csv_file_path: str, output_file: str = None):
        """CSV dosyasından JSON'a çevir"""
        try:
            # CSV dosyasını oku
            df = pd.read_csv(csv_file_path, encoding='utf-8')
            
            # Gerekli sütunları kontrol et
            required_columns = ['message', 'sender', 'timestamp']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Eksik sütunlar: {missing_columns}")
                print("📋 Gerekli sütunlar: message, sender, timestamp")
                print("📋 İsteğe bağlı: user_type, id")
                return None
            
            # JSON formatına çevir
            messages = []
            for i, row in df.iterrows():
                message = {
                    "id": row.get('id', i + 1),
                    "timestamp": str(row['timestamp']),
                    "sender": str(row['sender']),
                    "user_type": row.get('user_type', 'customer'),
                    "message": str(row['message'])
                }
                messages.append(message)
            
            conversation_data = {
                "conversation_id": f"chat_data_{datetime.now().strftime('%Y%m%d')}",
                "date": datetime.now().isoformat(),
                "total_messages": len(messages),
                "messages": messages
            }
            
            # JSON dosyasına kaydet
            if output_file is None:
                output_file = csv_file_path.replace('.csv', '.json')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ CSV verisi JSON'a çevrildi: {output_file}")
            return output_file
            
        except Exception as e:
            print(f"❌ Hata: {e}")
            return None
    
    def whatsapp_to_json(self, whatsapp_txt_path: str, output_file: str = None):
        """WhatsApp chat export'undan JSON'a çevir"""
        try:
            with open(whatsapp_txt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            messages = []
            message_id = 1
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # WhatsApp format: [DD/MM/YYYY, HH:MM:SS] Sender: Message
                if line.startswith('[') and ']' in line and ':' in line:
                    try:
                        # Timestamp ve sender'ı ayır
                        timestamp_end = line.find(']')
                        timestamp_str = line[1:timestamp_end]
                        
                        rest = line[timestamp_end + 2:]  # "] " kısmını atla
                        
                        if ':' in rest:
                            sender_end = rest.find(':')
                            sender = rest[:sender_end].strip()
                            message_text = rest[sender_end + 1:].strip()
                            
                            # User type'ı belirle (basit heuristic)
                            user_type = 'support' if any(word in sender.lower() for word in ['destek', 'admin', 'bot']) else 'customer'
                            
                            message = {
                                "id": message_id,
                                "timestamp": timestamp_str,
                                "sender": sender,
                                "user_type": user_type,
                                "message": message_text
                            }
                            messages.append(message)
                            message_id += 1
                    except:
                        continue
            
            conversation_data = {
                "conversation_id": f"whatsapp_chat_{datetime.now().strftime('%Y%m%d')}",
                "date": datetime.now().isoformat(),
                "total_messages": len(messages),
                "messages": messages
            }
            
            # JSON dosyasına kaydet
            if output_file is None:
                output_file = whatsapp_txt_path.replace('.txt', '.json')
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, ensure_ascii=False, indent=2)
            
            print(f"✅ WhatsApp verisi JSON'a çevrildi: {output_file}")
            print(f"📊 Toplam mesaj sayısı: {len(messages)}")
            return output_file
            
        except Exception as e:
            print(f"❌ Hata: {e}")
            return None
    
    def create_template_excel(self, filename: str = "chat_template.xlsx"):
        """Örnek Excel şablonu oluştur"""
        sample_data = {
            'id': [1, 2, 3, 4, 5],
            'timestamp': [
                '2024-01-15 10:30:00',
                '2024-01-15 10:32:00', 
                '2024-01-15 10:35:00',
                '2024-01-15 10:37:00',
                '2024-01-15 10:40:00'
            ],
            'sender': ['müşteri_1', 'destek_1', 'müşteri_1', 'destek_1', 'müşteri_2'],
            'user_type': ['customer', 'support', 'customer', 'support', 'customer'],
            'message': [
                'Merhaba, düğün mekanı arıyorum',
                'Merhaba! Hangi bölgede arıyorsunuz?',
                'İstanbul Avrupa yakası olsun',
                'Size uygun seçenekleri gösterebilirim',
                'Gelinlik modelleri var mı?'
            ]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_excel(filename, index=False)
        print(f"✅ Örnek Excel şablonu oluşturuldu: {filename}")
        return filename

def main():
    """Ana fonksiyon"""
    converter = ChatDataConverter()
    
    print("📁 Chat Verisi JSON Converter")
    print("="*40)
    print("1. Excel'den JSON'a çevir")
    print("2. CSV'den JSON'a çevir") 
    print("3. WhatsApp export'undan JSON'a çevir")
    print("4. Örnek Excel şablonu oluştur")
    print("5. Test verisi oluştur")
    
    choice = input("\nSeçiminiz (1-5): ").strip()
    
    if choice == "1":
        file_path = input("Excel dosya yolu: ").strip()
        if os.path.exists(file_path):
            converter.excel_to_json(file_path)
        else:
            print("❌ Dosya bulunamadı!")
    
    elif choice == "2":
        file_path = input("CSV dosya yolu: ").strip()
        if os.path.exists(file_path):
            converter.csv_to_json(file_path)
        else:
            print("❌ Dosya bulunamadı!")
    
    elif choice == "3":
        file_path = input("WhatsApp txt dosya yolu: ").strip()
        if os.path.exists(file_path):
            converter.whatsapp_to_json(file_path)
        else:
            print("❌ Dosya bulunamadı!")
    
    elif choice == "4":
        filename = input("Şablon dosya adı (varsayılan: chat_template.xlsx): ").strip()
        if not filename:
            filename = "chat_template.xlsx"
        converter.create_template_excel(filename)
    
    elif choice == "5":
        os.system("python test_data_generator.py")
    
    else:
        print("❌ Geçersiz seçim!")

if __name__ == "__main__":
    main()