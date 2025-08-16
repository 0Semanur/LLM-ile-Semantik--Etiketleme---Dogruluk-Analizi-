import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from matplotlib import rcParams

# Türkçe karakter desteği
rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.unicode_minus'] = False

class ChatAnalysisVisualizer:
    def __init__(self, data_source):
        """
        data_source: CSV dosya yolu, SQLite DB yolu veya DataFrame
        """
        if isinstance(data_source, str):
            if data_source.endswith('.csv'):
                self.df = pd.read_csv(data_source, encoding='utf-8-sig')
            elif data_source.endswith('.db'):
                conn = sqlite3.connect(data_source)
                self.df = pd.read_sql_query("SELECT * FROM chat_analysis", conn)
                conn.close()
        elif isinstance(data_source, pd.DataFrame):
            self.df = data_source
        else:
            raise ValueError("Desteklenmeyen veri formatı")
    
    def plot_sentiment_distribution(self):
        """Duygu dağılımı grafiği"""
        plt.figure(figsize=(10, 6))
        sentiment_counts = self.df['sentiment'].value_counts()
        
        colors = ['#2ecc71', '#e74c3c', '#95a5a6']  # Yeşil, Kırmızı, Gri
        plt.pie(sentiment_counts.values, labels=sentiment_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90)
        
        plt.title('Müşteri Mesajları Duygu Dağılımı', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('sentiment_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_category_distribution(self):
        """Kategori dağılımı grafiği"""
        plt.figure(figsize=(12, 8))
        category_counts = self.df['kategori'].value_counts()
        
        sns.barplot(x=category_counts.values, y=category_counts.index, palette='viridis')
        plt.title('Müşteri Sorularının Kategori Dağılımı', fontsize=16, fontweight='bold')
        plt.xlabel('Mesaj Sayısı', fontsize=12)
        plt.ylabel('Kategori', fontsize=12)
        
        # Değerleri çubukların üzerine yaz
        for i, v in enumerate(category_counts.values):
            plt.text(v + 0.1, i, str(v), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('category_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_unanswered_questions(self):
        """Yanıtlanmamış sorular analizi"""
        plt.figure(figsize=(10, 6))
        
        answered_counts = self.df['yanıtlanmış_mı'].value_counts()
        colors = ['#e74c3c', '#2ecc71']  # Kırmızı (Hayır), Yeşil (Evet)
        
        plt.pie(answered_counts.values, labels=answered_counts.index, 
                autopct='%1.1f%%', colors=colors, startangle=90)
        
        plt.title('Müşteri Sorularının Yanıtlanma Durumu', fontsize=16, fontweight='bold')
        plt.axis('equal')
        plt.tight_layout()
        plt.savefig('unanswered_questions.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_intent_analysis(self):
        """Amaç (intent) analizi"""
        plt.figure(figsize=(12, 8))
        intent_counts = self.df['intent'].value_counts()
        
        sns.barplot(x=intent_counts.values, y=intent_counts.index, palette='Set2')
        plt.title('Müşteri Mesajlarının Amaç Dağılımı', fontsize=16, fontweight='bold')
        plt.xlabel('Mesaj Sayısı', fontsize=12)
        plt.ylabel('Amaç (Intent)', fontsize=12)
        
        for i, v in enumerate(intent_counts.values):
            plt.text(v + 0.1, i, str(v), va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('intent_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self):
        """Kapsamlı görsel rapor oluştur"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('DüğünBuketi Müşteri Konuşmaları Analiz Raporu', 
                     fontsize=18, fontweight='bold')
        
        # 1. Duygu Dağılımı
        sentiment_counts = self.df['sentiment'].value_counts()
        axes[0, 0].pie(sentiment_counts.values, labels=sentiment_counts.index, 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 0].set_title('Duygu Dağılımı')
        
        # 2. Yanıtlanma Durumu
        answered_counts = self.df['yanıtlanmış_mı'].value_counts()
        axes[0, 1].pie(answered_counts.values, labels=answered_counts.index, 
                       autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('Yanıtlanma Durumu')
        
        # 3. En Çok Sorulan Kategoriler (Top 5)
        top_categories = self.df['kategori'].value_counts().head(5)
        axes[1, 0].bar(range(len(top_categories)), top_categories.values)
        axes[1, 0].set_xticks(range(len(top_categories)))
        axes[1, 0].set_xticklabels(top_categories.index, rotation=45, ha='right')
        axes[1, 0].set_title('En Çok Sorulan Kategoriler (Top 5)')
        axes[1, 0].set_ylabel('Mesaj Sayısı')
        
        # 4. En Çok Görülen Amaçlar (Top 5)
        top_intents = self.df['intent'].value_counts().head(5)
        axes[1, 1].bar(range(len(top_intents)), top_intents.values)
        axes[1, 1].set_xticks(range(len(top_intents)))
        axes[1, 1].set_xticklabels(top_intents.index, rotation=45, ha='right')
        axes[1, 1].set_title('En Çok Görülen Amaçlar (Top 5)')
        axes[1, 1].set_ylabel('Mesaj Sayısı')
        
        plt.tight_layout()
        plt.savefig('comprehensive_report.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_statistics(self):
        """İstatistiksel özet oluştur"""
        stats = {
            'Toplam Mesaj': len(self.df),
            'Yanıtlanmamış Soru': len(self.df[self.df['yanıtlanmış_mı'] == 'Hayır']),
            'Yanıtlanma Oranı': f"{(len(self.df[self.df['yanıtlanmış_mı'] == 'Evet']) / len(self.df) * 100):.1f}%",
            'En Çok Sorulan Kategori': self.df['kategori'].mode()[0],
            'En Yaygın Duygu': self.df['sentiment'].mode()[0],
            'En Yaygın Amaç': self.df['intent'].mode()[0]
        }
        
        return stats

# Kullanım örneği
if __name__ == "__main__":
    # CSV dosyasından görselleştirme
    # visualizer = ChatAnalysisVisualizer('dugum_buketi_analiz.csv')
    
    # SQLite'dan görselleştirme
    # visualizer = ChatAnalysisVisualizer('dugum_buketi_analiz.db')
    
    # visualizer.plot_sentiment_distribution()
    # visualizer.plot_category_distribution()
    # visualizer.plot_unanswered_questions()
    # visualizer.plot_intent_analysis()
    # visualizer.create_comprehensive_report()
    
    # stats = visualizer.generate_statistics()
    # print("İstatistiksel Özet:", stats)
    
    print("Görselleştirme modülü hazır!")