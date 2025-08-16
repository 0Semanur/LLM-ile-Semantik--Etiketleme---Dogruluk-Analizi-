import pandas as pd
import streamlit as st
import json
from datetime import datetime
import os

class ManualLabelingSystem:
    def __init__(self):
        """Manuel etiketleme sistemi"""
        self.sentiment_options = ["Pozitif", "Negatif", "Nötr"]
        self.topic_options = [
            "Düğün mekanı", "Gelinlik", "Fotoğrafçı", "Müzik/DJ", 
            "Çiçek/Dekorasyon", "Davetiye", "Pasta/Catering", 
            "Video çekimi", "Nikah şekeri", "Takı/Aksesuar",
            "Genel bilgi", "Fiyat sorgusu", "Rezervasyon", "Diğer"
        ]
        self.bot_response_options = ["Evet", "Hayır"]
    
    def create_labeling_interface(self):
        """Streamlit ile etiketleme arayüzü oluştur"""
        st.title("🏷️ Manuel Sohbet Etiketleme Sistemi")
        st.write("LLM sonuçlarını manuel olarak doğrulamak için bu arayüzü kullanın.")
        
        # Dosya yükleme
        uploaded_file = st.file_uploader(
            "LLM analiz sonuçları CSV dosyasını yükleyin",
            type=['csv']
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            
            # Session state'de veri sakla
            if 'df' not in st.session_state:
                st.session_state.df = df
                st.session_state.current_index = 0
                st.session_state.manual_labels = {}
            
            # İlerleme göstergesi
            total_messages = len(st.session_state.df)
            current_idx = st.session_state.current_index
            
            st.progress((current_idx + 1) / total_messages)
            st.write(f"Mesaj {current_idx + 1} / {total_messages}")
            
            if current_idx < total_messages:
                current_row = st.session_state.df.iloc[current_idx]
                
                # Mesaj bilgilerini göster
                st.subheader("📝 Analiz Edilecek Mesaj")
                st.write(f"**Gönderen:** {current_row['sender']}")
                st.write(f"**Tarih:** {current_row['timestamp']}")
                st.text_area("**Mesaj İçeriği:**", current_row['message'], height=100, disabled=True)
                
                # LLM tahminlerini göster
                st.subheader("🤖 LLM Tahminleri")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.write(f"**Duygu:** {current_row['llm_sentiment']}")
                with col2:
                    st.write(f"**Konu:** {current_row['llm_topic']}")
                with col3:
                    st.write(f"**Bot Yanıtı:** {current_row['llm_bot_response']}")
                
                # Manuel etiketleme
                st.subheader("✋ Manuel Etiketleme")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    manual_sentiment = st.selectbox(
                        "Doğru Duygu:",
                        self.sentiment_options,
                        index=self.sentiment_options.index(current_row['llm_sentiment'])
                    )
                
                with col2:
                    manual_topic = st.selectbox(
                        "Doğru Konu:",
                        self.topic_options,
                        index=self.topic_options.index(current_row['llm_topic'])
                    )
                
                with col3:
                    manual_bot_response = st.selectbox(
                        "Bot Yanıt Verdi mi?:",
                        self.bot_response_options,
                        index=self.bot_response_options.index(current_row['llm_bot_response'])
                    )
                
                # Kaydet ve devam et
                col1, col2, col3 = st.columns([1, 1, 1])
                
                with col1:
                    if st.button("⬅️ Önceki"):
                        if current_idx > 0:
                            st.session_state.current_index -= 1
                            st.rerun()
                
                with col2:
                    if st.button("💾 Kaydet ve Devam"):
                        # Manuel etiketleri kaydet
                        st.session_state.manual_labels[current_idx] = {
                            'manual_sentiment': manual_sentiment,
                            'manual_topic': manual_topic,
                            'manual_bot_response': manual_bot_response
                        }
                        
                        if current_idx < total_messages - 1:
                            st.session_state.current_index += 1
                            st.rerun()
                        else:
                            st.success("Tüm mesajlar etiketlendi!")
                
                with col3:
                    if st.button("➡️ Sonraki (Kaydetmeden)"):
                        if current_idx < total_messages - 1:
                            st.session_state.current_index += 1
                            st.rerun()
                
                # Sonuçları kaydet
                if st.session_state.manual_labels:
                    st.subheader("💾 Sonuçları Kaydet")
                    
                    if st.button("📁 Manuel Etiketleri CSV'ye Kaydet"):
                        self.save_manual_labels()
                        st.success("Manuel etiketler kaydedildi!")
            
            else:
                st.success("🎉 Tüm mesajlar etiketlendi!")
                if st.button("📁 Sonuçları Kaydet"):
                    self.save_manual_labels()
                    st.success("Sonuçlar kaydedildi!")
    
    def save_manual_labels(self):
        """Manuel etiketleri kaydet"""
        if 'manual_labels' in st.session_state and st.session_state.manual_labels:
            # DataFrame'i güncelle
            df = st.session_state.df.copy()
            
            for idx, labels in st.session_state.manual_labels.items():
                df.loc[idx, 'manual_sentiment'] = labels['manual_sentiment']
                df.loc[idx, 'manual_topic'] = labels['manual_topic']
                df.loc[idx, 'manual_bot_response'] = labels['manual_bot_response']
            
            # Dosyaya kaydet
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"manuel_etiketli_veri_{timestamp}.csv"
            filepath = os.path.join(os.getcwd(), filename)
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            
            st.download_button(
                label="📥 Dosyayı İndir",
                data=df.to_csv(index=False, encoding='utf-8-sig'),
                file_name=filename,
                mime='text/csv'
            )
            
            return filepath
        
        return None

def run_labeling_app():
    """Etiketleme uygulamasını çalıştır"""
    labeling_system = ManualLabelingSystem()
    labeling_system.create_labeling_interface()

if __name__ == "__main__":
    run_labeling_app()