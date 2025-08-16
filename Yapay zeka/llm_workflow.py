import os
from dotenv import load_dotenv
from llm_analyzer import LLMChatAnalyzer
from accuracy_analyzer import AccuracyAnalyzer
import json
import pandas as pd
from datetime import datetime

class LLMWorkflow:
    def __init__(self, provider="openai"):
        """LLM iş akışı koordinatörü"""
        load_dotenv()
        self.llm_analyzer = LLMChatAnalyzer(provider=provider)
        self.accuracy_analyzer = AccuracyAnalyzer()
        
    def run_full_analysis(self, chat_data_path: str):
        """Tam analiz iş akışını çalıştır"""
        print("🚀 LLM Analiz İş Akışı Başlatılıyor...")
        
        # 1. Chat verisini yükle
        print("📂 Chat verisi yükleniyor...")
        with open(chat_data_path, 'r', encoding='utf-8') as f:
            chat_data = json.load(f)
        
        # 2. LLM ile analiz yap
        print("🤖 LLM analizi başlatılıyor...")
        llm_results = self.llm_analyzer.analyze_conversation_llm(chat_data)
        
        # 3. LLM sonuçlarını kaydet
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        llm_file = f"llm_analiz_{timestamp}.csv"
        self.llm_analyzer.save_llm_results(llm_results, llm_file)
        
        print(f"✅ LLM analizi tamamlandı: {llm_file}")
        print("\n📋 Sonraki adımlar:")
        print("1. Manuel etiketleme için şu komutu çalıştırın:")
        print(f"   streamlit run manual_labeling.py")
        print("2. Açılan arayüzde LLM sonuçlarını manuel olarak doğrulayın")
        print("3. Manuel etiketli dosyayı kaydedin")
        print("4. Doğruluk analizi için bu scripti tekrar çalıştırın")
        
        return llm_file
    
    def run_accuracy_analysis(self, manual_labeled_file: str):
        """Manuel etiketli veri ile doğruluk analizi yap"""
        print("📊 Doğruluk analizi başlatılıyor...")
        
        # Veriyi yükle
        df = self.accuracy_analyzer.load_data(manual_labeled_file)
        
        # Analiz yap ve kaydet
        output_dir = self.accuracy_analyzer.save_detailed_analysis(df)
        
        # Raporu yazdır
        report = self.accuracy_analyzer.create_accuracy_report(df)
        print("\n" + "="*50)
        print(report)
        print("="*50)
        
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

def main():
    """Ana fonksiyon"""
    print("🎯 LLM Doğruluk Analizi Sistemi")
    print("="*40)
    
    # Workflow oluştur
    workflow = LLMWorkflow(provider="openai")  # veya "anthropic"
    
    # Kullanıcıdan seçim al
    print("\nNe yapmak istiyorsunuz?")
    print("1. Yeni chat verisi analiz et (LLM)")
    print("2. Manuel etiketli veri ile doğruluk analizi yap")
    print("3. Hedef başarı kontrolü")
    
    choice = input("\nSeçiminiz (1/2/3): ").strip()
    
    if choice == "1":
        chat_file = input("Chat verisi JSON dosya yolu: ").strip()
        if os.path.exists(chat_file):
            llm_file = workflow.run_full_analysis(chat_file)
        else:
            print("❌ Dosya bulunamadı!")
    
    elif choice == "2":
        manual_file = input("Manuel etiketli CSV dosya yolu: ").strip()
        if os.path.exists(manual_file):
            output_dir = workflow.run_accuracy_analysis(manual_file)
            print(f"\n📁 Analiz sonuçları: {output_dir}")
        else:
            print("❌ Dosya bulunamadı!")
    
    elif choice == "3":
        manual_file = input("Manuel etiketli CSV dosya yolu: ").strip()
        if os.path.exists(manual_file):
            results = workflow.check_target_achievement(manual_file)
            
            print("\n🎯 HEDEF BAŞARI RAPORU")
            print("="*30)
            
            for category, result in results.items():
                category_name = {
                    'sentiment': 'Sentiment',
                    'topic': 'Konu', 
                    'bot_response': 'Bot Yanıt'
                }[category]
                
                status = "✅ BAŞARILI" if result['target_achieved'] else "❌ BAŞARISIZ"
                print(f"{category_name}: %{result['accuracy']*100:.2f} - {status}")
                
                if not result['target_achieved']:
                    print(f"  Hedefe kalan: %{result['gap']*100:.2f}")
        else:
            print("❌ Dosya bulunamadı!")

if __name__ == "__main__":
    main()