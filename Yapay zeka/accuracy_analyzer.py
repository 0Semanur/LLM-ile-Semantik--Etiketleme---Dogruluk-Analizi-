import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
from datetime import datetime
import os

class AccuracyAnalyzer:
    def __init__(self):
        """Doğruluk analizi sistemi"""
        self.metrics = {}
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """Manuel etiketli veriyi yükle"""
        df = pd.read_csv(filepath)
        
        # Gerekli sütunları kontrol et
        required_columns = [
            'llm_sentiment', 'manual_sentiment',
            'llm_topic', 'manual_topic', 
            'llm_bot_response', 'manual_bot_response'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Eksik sütunlar: {missing_columns}")
        
        # NaN değerleri olan satırları filtrele
        print(f"📊 Toplam satır sayısı: {len(df)}")
        
        # Manuel etiketleme yapılmış satırları filtrele
        df_clean = df.dropna(subset=['manual_sentiment', 'manual_topic', 'manual_bot_response'])
        
        print(f"📊 Manuel etiketlenmiş satır sayısı: {len(df_clean)}")
        print(f"📊 Etiketlenmemiş satır sayısı: {len(df) - len(df_clean)}")
        
        if len(df_clean) == 0:
            raise ValueError("Hiç manuel etiketlenmiş veri bulunamadı!")
        
        return df_clean
    
    def calculate_accuracy_metrics(self, df: pd.DataFrame) -> dict:
        """Her kategori için doğruluk metriklerini hesapla"""
        metrics = {}
        
        # Veri temizliği - NaN değerleri kontrol et
        df_clean = df.dropna(subset=['manual_sentiment', 'manual_topic', 'manual_bot_response', 
                                   'llm_sentiment', 'llm_topic', 'llm_bot_response'])
        
        if len(df_clean) == 0:
            raise ValueError("Analiz için yeterli temiz veri yok!")
        
        print(f"📊 Analiz edilen temiz veri sayısı: {len(df_clean)}")
        
        # Sentiment analizi metrikleri
        sentiment_accuracy = accuracy_score(df_clean['manual_sentiment'], df_clean['llm_sentiment'])
        sentiment_precision = precision_score(df_clean['manual_sentiment'], df_clean['llm_sentiment'], average='weighted', zero_division=0)
        sentiment_recall = recall_score(df_clean['manual_sentiment'], df_clean['llm_sentiment'], average='weighted', zero_division=0)
        sentiment_f1 = f1_score(df_clean['manual_sentiment'], df_clean['llm_sentiment'], average='weighted', zero_division=0)
        
        metrics['sentiment'] = {
            'accuracy': sentiment_accuracy,
            'precision': sentiment_precision,
            'recall': sentiment_recall,
            'f1_score': sentiment_f1
        }
        
        # Konu analizi metrikleri
        topic_accuracy = accuracy_score(df_clean['manual_topic'], df_clean['llm_topic'])
        topic_precision = precision_score(df_clean['manual_topic'], df_clean['llm_topic'], average='weighted', zero_division=0)
        topic_recall = recall_score(df_clean['manual_topic'], df_clean['llm_topic'], average='weighted', zero_division=0)
        topic_f1 = f1_score(df_clean['manual_topic'], df_clean['llm_topic'], average='weighted', zero_division=0)
        
        metrics['topic'] = {
            'accuracy': topic_accuracy,
            'precision': topic_precision,
            'recall': topic_recall,
            'f1_score': topic_f1
        }
        
        # Bot yanıt analizi metrikleri
        bot_accuracy = accuracy_score(df_clean['manual_bot_response'], df_clean['llm_bot_response'])
        bot_precision = precision_score(df_clean['manual_bot_response'], df_clean['llm_bot_response'], average='weighted', zero_division=0)
        bot_recall = recall_score(df_clean['manual_bot_response'], df_clean['llm_bot_response'], average='weighted', zero_division=0)
        bot_f1 = f1_score(df_clean['manual_bot_response'], df_clean['llm_bot_response'], average='weighted', zero_division=0)
        
        metrics['bot_response'] = {
            'accuracy': bot_accuracy,
            'precision': bot_precision,
            'recall': bot_recall,
            'f1_score': bot_f1
        }
        
        self.metrics = metrics
        return metrics
    
    def generate_confusion_matrices(self, df: pd.DataFrame) -> dict:
        """Karışıklık matrislerini oluştur"""
        confusion_matrices = {}
        
        # Veri temizliği
        df_clean = df.dropna(subset=['manual_sentiment', 'manual_topic', 'manual_bot_response', 
                                   'llm_sentiment', 'llm_topic', 'llm_bot_response'])
        
        # Sentiment confusion matrix
        sentiment_cm = confusion_matrix(df_clean['manual_sentiment'], df_clean['llm_sentiment'])
        sentiment_labels = sorted(df_clean['manual_sentiment'].unique())
        confusion_matrices['sentiment'] = {
            'matrix': sentiment_cm,
            'labels': sentiment_labels
        }
        
        # Topic confusion matrix
        topic_cm = confusion_matrix(df_clean['manual_topic'], df_clean['llm_topic'])
        topic_labels = sorted(df_clean['manual_topic'].unique())
        confusion_matrices['topic'] = {
            'matrix': topic_cm,
            'labels': topic_labels
        }
        
        # Bot response confusion matrix
        bot_cm = confusion_matrix(df_clean['manual_bot_response'], df_clean['llm_bot_response'])
        bot_labels = sorted(df_clean['manual_bot_response'].unique())
        confusion_matrices['bot_response'] = {
            'matrix': bot_cm,
            'labels': bot_labels
        }
        
        return confusion_matrices
    
    def create_accuracy_report(self, df: pd.DataFrame) -> str:
        """Detaylı doğruluk raporu oluştur"""
        # Veri temizliği
        df_clean = df.dropna(subset=['manual_sentiment', 'manual_topic', 'manual_bot_response', 
                                   'llm_sentiment', 'llm_topic', 'llm_bot_response'])
        
        metrics = self.calculate_accuracy_metrics(df)
        
        report = f"""
# 📊 LLM Doğruluk Analizi Raporu
**Tarih:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Toplam Mesaj:** {len(df)}
**Analiz Edilen Mesaj:** {len(df_clean)}
**Etiketlenmemiş Mesaj:** {len(df) - len(df_clean)}

## 🎯 Genel Doğruluk Oranları

### 1. Sentiment Analizi
- **Doğruluk Oranı:** %{metrics['sentiment']['accuracy']*100:.2f}
- **Precision:** %{metrics['sentiment']['precision']*100:.2f}
- **Recall:** %{metrics['sentiment']['recall']*100:.2f}
- **F1-Score:** %{metrics['sentiment']['f1_score']*100:.2f}

### 2. Konu Analizi
- **Doğruluk Oranı:** %{metrics['topic']['accuracy']*100:.2f}
- **Precision:** %{metrics['topic']['precision']*100:.2f}
- **Recall:** %{metrics['topic']['recall']*100:.2f}
- **F1-Score:** %{metrics['topic']['f1_score']*100:.2f}

### 3. Bot Yanıt Analizi
- **Doğruluk Oranı:** %{metrics['bot_response']['accuracy']*100:.2f}
- **Precision:** %{metrics['bot_response']['precision']*100:.2f}
- **Recall:** %{metrics['bot_response']['recall']*100:.2f}
- **F1-Score:** %{metrics['bot_response']['f1_score']*100:.2f}

## 📈 Hedef Analizi
**%95 Doğruluk Hedefi:**
"""
        
        # Hedef analizi
        target_accuracy = 0.95
        
        for category, metric in metrics.items():
            accuracy = metric['accuracy']
            if accuracy >= target_accuracy:
                status = "✅ HEDEFİ AŞTI"
            else:
                gap = (target_accuracy - accuracy) * 100
                status = f"❌ Hedefe {gap:.2f}% kaldı"
            
            category_name = {
                'sentiment': 'Sentiment',
                'topic': 'Konu',
                'bot_response': 'Bot Yanıt'
            }[category]
            
            report += f"\n- **{category_name}:** %{accuracy*100:.2f} - {status}"
        
        # Öneriler
        report += "\n\n## 💡 İyileştirme Önerileri\n"
        
        for category, metric in metrics.items():
            if metric['accuracy'] < target_accuracy:
                category_name = {
                    'sentiment': 'Sentiment analizi',
                    'topic': 'Konu sınıflandırması', 
                    'bot_response': 'Bot yanıt tespiti'
                }[category]
                
                report += f"\n### {category_name} için:\n"
                report += "- Prompt mühendisliği geliştirin\n"
                report += "- Daha fazla örnek veri ekleyin\n"
                report += "- Farklı LLM modeli deneyin\n"
        
        return report
    
    def create_visualizations(self, df: pd.DataFrame, save_path: str = None):
        """Görselleştirmeler oluştur"""
        # Veri temizliği
        df_clean = df.dropna(subset=['manual_sentiment', 'manual_topic', 'manual_bot_response', 
                                   'llm_sentiment', 'llm_topic', 'llm_bot_response'])
        
        metrics = self.calculate_accuracy_metrics(df)
        confusion_matrices = self.generate_confusion_matrices(df)
        
        # 1. Doğruluk oranları bar chart
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Doğruluk Oranları', 'Precision Skorları', 'Recall Skorları', 'F1 Skorları'),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        categories = ['Sentiment', 'Konu', 'Bot Yanıt']
        accuracy_values = [metrics['sentiment']['accuracy'], metrics['topic']['accuracy'], metrics['bot_response']['accuracy']]
        precision_values = [metrics['sentiment']['precision'], metrics['topic']['precision'], metrics['bot_response']['precision']]
        recall_values = [metrics['sentiment']['recall'], metrics['topic']['recall'], metrics['bot_response']['recall']]
        f1_values = [metrics['sentiment']['f1_score'], metrics['topic']['f1_score'], metrics['bot_response']['f1_score']]
        
        # Doğruluk oranları
        fig.add_trace(
            go.Bar(x=categories, y=[v*100 for v in accuracy_values], name='Doğruluk'),
            row=1, col=1
        )
        
        # Precision
        fig.add_trace(
            go.Bar(x=categories, y=[v*100 for v in precision_values], name='Precision'),
            row=1, col=2
        )
        
        # Recall
        fig.add_trace(
            go.Bar(x=categories, y=[v*100 for v in recall_values], name='Recall'),
            row=2, col=1
        )
        
        # F1 Score
        fig.add_trace(
            go.Bar(x=categories, y=[v*100 for v in f1_values], name='F1 Score'),
            row=2, col=2
        )
        
        # %95 hedef çizgisi ekle
        for row in range(1, 3):
            for col in range(1, 3):
                fig.add_hline(y=95, line_dash="dash", line_color="red", 
                             annotation_text="Hedef %95", row=row, col=col)
        
        fig.update_layout(
            title_text="LLM Performans Metrikleri",
            showlegend=False,
            height=600
        )
        
        # Y ekseni %100'e kadar
        fig.update_yaxes(range=[0, 100])
        
        if save_path:
            fig.write_html(f"{save_path}/performance_metrics.html")
            fig.write_image(f"{save_path}/performance_metrics.png")
        
        fig.show()
        
        # 2. Confusion Matrix görselleştirmeleri
        self.plot_confusion_matrices(confusion_matrices, save_path)
        
        return fig
    
    def plot_confusion_matrices(self, confusion_matrices: dict, save_path: str = None):
        """Karışıklık matrislerini görselleştir"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        titles = ['Sentiment Analizi', 'Konu Sınıflandırması', 'Bot Yanıt Tespiti']
        categories = ['sentiment', 'topic', 'bot_response']
        
        for i, (category, title) in enumerate(zip(categories, titles)):
            cm_data = confusion_matrices[category]
            
            sns.heatmap(
                cm_data['matrix'], 
                annot=True, 
                fmt='d',
                xticklabels=cm_data['labels'],
                yticklabels=cm_data['labels'],
                ax=axes[i],
                cmap='Blues'
            )
            
            axes[i].set_title(title)
            axes[i].set_xlabel('LLM Tahmini')
            axes[i].set_ylabel('Manuel Etiket')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(f"{save_path}/confusion_matrices.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_detailed_analysis(self, df: pd.DataFrame, output_dir: str = None):
        """Detaylı analizi kaydet"""
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"accuracy_analysis_{timestamp}"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Rapor kaydet
        report = self.create_accuracy_report(df)
        with open(f"{output_dir}/accuracy_report.md", 'w', encoding='utf-8') as f:
            f.write(report)
        
        # Metrikleri JSON olarak kaydet
        with open(f"{output_dir}/metrics.json", 'w', encoding='utf-8') as f:
            json.dump(self.metrics, f, indent=2, ensure_ascii=False)
        
        # Görselleştirmeleri kaydet
        self.create_visualizations(df, output_dir)
        
        # Detaylı veri analizi
        detailed_analysis = self.create_detailed_analysis(df)
        detailed_analysis.to_csv(f"{output_dir}/detailed_analysis.csv", index=False, encoding='utf-8-sig')
        
        print(f"Detaylı analiz kaydedildi: {output_dir}")
        return output_dir
    
    def create_detailed_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Her mesaj için detaylı analiz"""
        # Veri temizliği
        analysis_df = df.dropna(subset=['manual_sentiment', 'manual_topic', 'manual_bot_response', 
                                      'llm_sentiment', 'llm_topic', 'llm_bot_response']).copy()
        
        # Doğruluk sütunları ekle
        analysis_df['sentiment_correct'] = analysis_df['manual_sentiment'] == analysis_df['llm_sentiment']
        analysis_df['topic_correct'] = analysis_df['manual_topic'] == analysis_df['llm_topic']
        analysis_df['bot_response_correct'] = analysis_df['manual_bot_response'] == analysis_df['llm_bot_response']
        
        # Genel doğruluk skoru
        analysis_df['overall_correct'] = (
            analysis_df['sentiment_correct'] & 
            analysis_df['topic_correct'] & 
            analysis_df['bot_response_correct']
        )
        
        return analysis_df

def main():
    """Ana fonksiyon - örnek kullanım"""
    analyzer = AccuracyAnalyzer()
    
    # Örnek veri yükle (gerçek dosya yolunu kullanın)
    # df = analyzer.load_data("manuel_etiketli_veri_20241215_143022.csv")
    
    # Analiz yap
    # output_dir = analyzer.save_detailed_analysis(df)
    
    print("Doğruluk analizi tamamlandı!")

if __name__ == "__main__":
    main()