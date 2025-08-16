import os
import sys
from pathlib import Path
from enhanced_llm_analyzer import EnhancedLLMAnalyzer
from accuracy_analyzer import AccuracyAnalyzer
import json
import pandas as pd
from datetime import datetime
import logging

# Logging ayarla
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MainWorkflow:
    def __init__(self):
        """Ana iş akışı koordinatörü"""
        self.llm_analyzer = None
        self.accuracy_analyzer = AccuracyAnalyzer()
        
    def setup_llm_analyzer(self, provider="groq", model=None):
        """LLM analyzer'ı ayarla"""
        try:
            self.llm_analyzer = EnhancedLLMAnalyzer(provider=provider, model=model)
            logger.info(f"LLM Analyzer ayarlandı: {provider} - {self.llm_analyzer.model}")
            return True
        except Exception as e:
            logger.error(f"LLM Analyzer ayarlanamadı: {e}")
            return False
    
    def step1_llm_analysis(self, chat_data_path: str) -> str:
        """Adım 1: LLM ile sohbet analizi"""
        logger.info("🚀 Adım 1: LLM Analizi Başlatılıyor...")
        
        if not self.llm_analyzer:
            raise ValueError("LLM Analyzer ayarlanmamış!")
        
        # Chat verisini yükle
        logger.info(f"📂 Chat verisi yükleniyor: {chat_data_path}")
        
        if not os.path.exists(chat_data_path):
            raise FileNotFoundError(f"Chat verisi bulunamadı: {chat_data_path}")
        
        with open(chat_data_path, 'r', encoding='utf-8') as f:
            chat_data_raw = json.load(f)
        
        # JSON formatını kontrol et ve messages listesini çıkar
        if isinstance(chat_data_raw, dict) and 'messages' in chat_data_raw:
            chat_data = chat_data_raw['messages']
            logger.info(f"📊 JSON formatı tespit edildi. {len(chat_data)} mesaj bulundu.")
        elif isinstance(chat_data_raw, list):
            chat_data = chat_data_raw
            logger.info(f"📊 Liste formatı tespit edildi. {len(chat_data)} mesaj bulundu.")
        else:
            raise ValueError("Geçersiz JSON formatı! 'messages' anahtarı veya liste formatı bekleniyor.")
        
        # Mesaj formatını düzelt (id -> message_id)
        for message in chat_data:
            if 'id' in message and 'message_id' not in message:
                message['message_id'] = message['id']
        
        # LLM analizi yap
        logger.info("🤖 LLM analizi başlatılıyor...")
        results_df = self.llm_analyzer.analyze_conversation(chat_data)
        
        # Sonuçları kaydet
        csv_file, metadata_file = self.llm_analyzer.save_analysis_results(results_df)
        
        logger.info(f"✅ LLM analizi tamamlandı: {csv_file}")
        
        # Sonraki adım talimatları
        print("\n" + "="*60)
        print("🎯 ADIM 1 TAMAMLANDI!")
        print("="*60)
        print(f"📁 LLM analiz sonuçları: {csv_file}")
        print(f"📋 Metadata dosyası: {metadata_file}")
        print(f"🤖 Kullanılan Provider: {self.llm_analyzer.provider}")
        print(f"🧠 Kullanılan Model: {self.llm_analyzer.model}")
        print(f"🔄 API çağrısı sayısı: {self.llm_analyzer.api_calls}")
        print(f"🎯 Token kullanımı: {self.llm_analyzer.total_tokens}")
        print("\n📋 SONRAKI ADIMLAR:")
        print("1. Manuel etiketleme için şu komutu çalıştırın:")
        print("   streamlit run enhanced_manual_labeling.py")
        print("2. Açılan arayüzde LLM sonuçlarını manuel olarak doğrulayın")
        print("3. Manuel etiketli dosyayı indirin")
        print("4. Adım 3 için bu scripti tekrar çalıştırın")
        print("="*60)
        
        return csv_file
    
    def step2_manual_labeling_instructions(self):
        """Adım 2: Manuel etiketleme talimatları"""
        print("\n" + "="*60)
        print("🏷️ ADIM 2: MANUEL ETİKETLEME")
        print("="*60)
        print("1. Aşağıdaki komutu çalıştırın:")
        print("   streamlit run enhanced_manual_labeling.py")
        print("\n2. Açılan web arayüzünde:")
        print("   - LLM analiz sonuçları CSV dosyasını yükleyin")
        print("   - Her mesajı manuel olarak etiketleyin")
        print("   - Sentiment: Pozitif/Negatif/Nötr")
        print("   - Konu: DüğünBuketi kategorilerinden seçin")
        print("   - Bot Yanıtı: Evet/Hayır")
        print("\n3. Tüm etiketleme tamamlandıktan sonra:")
        print("   - 'Sonuçları İndir' butonuna tıklayın")
        print("   - İndirilen CSV dosyasını kaydedin")
        print("\n4. Adım 3 için bu scripti tekrar çalıştırın")
        print("="*60)
    
    def step3_accuracy_analysis(self, manual_labeled_file: str) -> str:
        """Adım 3: Doğruluk analizi"""
        logger.info("📊 Adım 3: Doğruluk Analizi Başlatılıyor...")
        
        if not os.path.exists(manual_labeled_file):
            raise FileNotFoundError(f"Manuel etiketli dosya bulunamadı: {manual_labeled_file}")
        
        # Veriyi yükle
        logger.info(f"📂 Manuel etiketli veri yükleniyor: {manual_labeled_file}")
        df = self.accuracy_analyzer.load_data(manual_labeled_file)
        
        # Analiz yap
        logger.info("🔍 Doğruluk analizi yapılıyor...")
        output_dir = self.accuracy_analyzer.save_detailed_analysis(df)
        
        # Hedef kontrolü
        target_results = self.check_target_achievement(manual_labeled_file)
        
        # Raporu göster
        report = self.accuracy_analyzer.create_accuracy_report(df)
        
        print("\n" + "="*60)
        print("🎯 ADIM 3 TAMAMLANDI!")
        print("="*60)
        print(report)
        print("\n📁 Detaylı analiz sonuçları:", output_dir)
        print("="*60)
        
        # Hedef başarı durumu
        self.print_target_status(target_results)
        
        return output_dir
    
    def check_target_achievement(self, manual_labeled_file: str) -> dict:
        """%95 hedefine ulaşılıp ulaşılmadığını kontrol et"""
        df = self.accuracy_analyzer.load_data(manual_labeled_file)
        metrics = self.accuracy_analyzer.calculate_accuracy_metrics(df)
        
        target = 0.95
        results = {}
        
        for category, metric in metrics.items():
            accuracy = metric['accuracy']
            results[category] = {
                'accuracy': accuracy,
                'target_achieved': accuracy >= target,
                'gap': max(0, target - accuracy)
            }
        
        return results
    
    def print_target_status(self, target_results: dict):
        """Hedef durumunu yazdır"""
        print("\n🎯 HEDEF BAŞARI DURUMU (%95)")
        print("="*40)
        
        all_achieved = True
        
        for category, result in target_results.items():
            category_name = {
                'sentiment': 'Sentiment',
                'topic': 'Konu',
                'bot_response': 'Bot Yanıt'
            }[category]
            
            accuracy_pct = result['accuracy'] * 100
            
            if result['target_achieved']:
                status = "✅ BAŞARILI"
                print(f"{category_name:12}: %{accuracy_pct:5.1f} - {status}")
            else:
                status = "❌ BAŞARISIZ"
                gap_pct = result['gap'] * 100
                print(f"{category_name:12}: %{accuracy_pct:5.1f} - {status} (Kalan: %{gap_pct:.1f})")
                all_achieved = False
        
        print("="*40)
        
        if all_achieved:
            print("🎉 TÜM HEDEFLER BAŞARILI!")
            print("Prompt mühendisliği başarılı, sistemi kullanabilirsiniz.")
        else:
            print("⚠️  Bazı hedefler başarısız!")
            print("Prompt mühendisliği geliştirme önerileri:")
            print("- Daha spesifik örnekler ekleyin")
            print("- Sistem mesajını geliştirin")
            print("- Farklı model deneyin")
            print("- Daha fazla training verisi kullanın")

def main():
    """Ana fonksiyon"""
    print("🎯 DüğünBuketi LLM Analiz Sistemi")
    print("="*50)
    
    workflow = MainWorkflow()
    
    while True:
        print("\nNe yapmak istiyorsunuz?")
        print("1. Adım 1: LLM ile sohbet analizi")
        print("2. Adım 2: Manuel etiketleme talimatları")
        print("3. Adım 3: Doğruluk analizi")
        print("4. Hedef başarı kontrolü")
        print("5. Test verisi oluştur")
        print("0. Çıkış")
        
        choice = input("\nSeçiminiz (0-5): ").strip()
        
        try:
            if choice == "1":
                # LLM provider seçimi
                print("\nLLM Provider seçin:")
                print("1. Groq (llama3-8b-8192) - ÜCRETSİZ ⚡")
                print("2. Groq (mixtral-8x7b-32768) - ÜCRETSİZ")
                print("3. Groq (llama3-70b-8192) - ÜCRETSİZ")
                print("4. Hugging Face (ücretsiz)")
                print("5. OpenAI (gpt-3.5-turbo)")
                print("6. OpenAI (gpt-4)")
                print("7. OpenAI (gpt-4o) - EN YENİ MODEL 🚀")
                
                provider_choice = input("Seçim (1-7): ").strip()
                
                if provider_choice == "1":
                    provider = "groq"
                    model = "llama3-8b-8192"
                elif provider_choice == "2":
                    provider = "groq"
                    model = "mixtral-8x7b-32768"
                elif provider_choice == "3":
                    provider = "groq"
                    model = "llama3-70b-8192"
                elif provider_choice == "4":
                    provider = "huggingface"
                    model = "microsoft/DialoGPT-medium"
                elif provider_choice == "5":
                    provider = "openai"
                    model = "gpt-3.5-turbo"
                elif provider_choice == "6":
                    provider = "openai"
                    model = "gpt-4"
                elif provider_choice == "7":
                    provider = "openai"
                    model = "gpt-4o"
                else:
                    print("❌ Geçersiz seçim!")
                    continue
                
                print(f"\n🤖 Seçilen Provider: {provider}")
                print(f"🧠 Seçilen Model: {model}")
                
                if not workflow.setup_llm_analyzer(provider, model):
                    print("❌ LLM Analyzer ayarlanamadı! API anahtarını kontrol edin.")
                    continue
                
                chat_file = input("\nChat verisi JSON dosya yolu: ").strip()
                
                if not chat_file:
                    chat_file = "sample_chat_data.json"
                    print(f"Varsayılan dosya kullanılıyor: {chat_file}")
                
                if os.path.exists(chat_file):
                    workflow.step1_llm_analysis(chat_file)
                else:
                    print("❌ Dosya bulunamadı!")
                    print("Test verisi oluşturmak için seçenek 5'i kullanın.")
            
            elif choice == "2":
                workflow.step2_manual_labeling_instructions()
            
            elif choice == "3":
                manual_file = input("Manuel etiketli CSV dosya yolu: ").strip()
                
                if os.path.exists(manual_file):
                    workflow.step3_accuracy_analysis(manual_file)
                else:
                    print("❌ Dosya bulunamadı!")
            
            elif choice == "4":
                manual_file = input("Manuel etiketli CSV dosya yolu: ").strip()
                
                if os.path.exists(manual_file):
                    results = workflow.check_target_achievement(manual_file)
                    workflow.print_target_status(results)
                else:
                    print("❌ Dosya bulunamadı!")
            
            elif choice == "5":
                print("Test verisi oluşturuluyor...")
                os.system("python test_data_generator.py")
                print("✅ Test verisi oluşturuldu: sample_chat_data.json")
            
            elif choice == "0":
                print("👋 Görüşürüz!")
                break
            
            else:
                print("❌ Geçersiz seçim!")
        
        except Exception as e:
            logger.error(f"Hata: {e}")
            print(f"❌ Hata oluştu: {e}")

if __name__ == "__main__":
    main()