import streamlit as st
import pandas as pd
import json
import time
from datetime import datetime
import os
import plotly.express as px
import plotly.graph_objects as go

class EnhancedManualLabeling:
    def __init__(self):
        """Gelişmiş manuel etiketleme sistemi"""
        
        # DüğünBuketi kategorileri
        self.dugum_buketi_categories = [
            "Düğün Mekanı", "Düğün Organizasyonu", "Gelinlik", "Fotoğrafçı",
            "Video Çekimi", "Müzik/DJ", "Çiçek/Dekorasyon", "Davetiye",
            "Pasta/Catering", "Nikah Şekeri", "Takı/Aksesuar", "Düğün Arabası",
            "Düğün Dansı", "Genel Bilgi", "Fiyat Sorgusu", "Rezervasyon",
            "Şikayet", "Diğer"
        ]
        
        self.sentiment_options = ["Pozitif", "Negatif", "Nötr"]
        self.bot_response_options = ["Evet", "Hayır"]
        
    def create_enhanced_interface(self):
        """Gelişmiş etiketleme arayüzü"""
        
        st.set_page_config(
            page_title="Manuel Etiketleme Sistemi",
            page_icon="🏷️",
            layout="wide"
        )
        
        st.title("🏷️ Gelişmiş Manuel Etiketleme Sistemi")
        st.markdown("---")
        
        # Sidebar - Dosya yükleme ve ayarlar
        with st.sidebar:
            st.header("📁 Dosya Yükleme")
            uploaded_file = st.file_uploader(
                "LLM analiz sonuçları CSV dosyasını yükleyin",
                type=['csv'],
                help="Enhanced LLM analyzer çıktısı olan CSV dosyasını seçin"
            )
            
            if uploaded_file is not None:
                df = pd.read_csv(uploaded_file)
                
                # Session state başlatma
                if 'df' not in st.session_state:
                    st.session_state.df = df
                    st.session_state.current_index = 0
                    st.session_state.manual_labels = {}
                    st.session_state.labeling_start_time = datetime.now()
                
                # İstatistikler
                st.header("📊 İstatistikler")
                total_messages = len(st.session_state.df)
                labeled_count = len(st.session_state.manual_labels)
                progress_pct = (labeled_count / total_messages) * 100
                
                st.metric("Toplam Mesaj", total_messages)
                st.metric("Etiketlenen", labeled_count)
                st.metric("İlerleme", f"%{progress_pct:.1f}")
                
                # İlerleme çubuğu
                st.progress(progress_pct / 100)
                
                # Hızlı navigasyon
                st.header("🧭 Navigasyon")
                jump_to = st.number_input(
                    "Mesaja git:",
                    min_value=1,
                    max_value=total_messages,
                    value=st.session_state.current_index + 1
                )
                
                if st.button("Git"):
                    st.session_state.current_index = jump_to - 1
                    st.rerun()
                
                # Manuel etiketli CSV kaydetme
                st.header("💾 Kaydetme")
                if st.button("📁 Manuel Etiketli CSV Kaydet"):
                    self.save_manual_labeled_csv()
        
        # Ana içerik
        if 'df' in st.session_state:
            self.render_labeling_interface()
        else:
            st.info("👆 Lütfen sol panelden CSV dosyasını yükleyin")
    
    def save_manual_labeled_csv(self):
        """Manuel etiketli CSV'yi kaydet"""
        df = st.session_state.df.copy()
        
        # Manuel etiketleri ekle
        for idx, labels in st.session_state.manual_labels.items():
            if idx < len(df):
                df.loc[idx, 'manual_sentiment'] = labels.get('manual_sentiment', '')
                df.loc[idx, 'manual_topic'] = labels.get('manual_topic', '')
                df.loc[idx, 'manual_bot_response'] = labels.get('manual_bot_response', '')
                df.loc[idx, 'labeled_at'] = labels.get('labeled_at', '')
                df.loc[idx, 'comment'] = labels.get('comment', '')
        
        # Etiketlenmemiş satırlar için boş değerler
        if 'manual_sentiment' not in df.columns:
            df['manual_sentiment'] = ''
        if 'manual_topic' not in df.columns:
            df['manual_topic'] = ''
        if 'manual_bot_response' not in df.columns:
            df['manual_bot_response'] = ''
        if 'labeled_at' not in df.columns:
            df['labeled_at'] = ''
        if 'comment' not in df.columns:
            df['comment'] = ''
        
        # Dosya adı oluştur
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manuel_etiketli_veri_{timestamp}.csv"
        
        # Kaydet
        df.to_csv(filename, index=False, encoding='utf-8-sig')
        
        st.success(f"✅ Manuel etiketli veri kaydedildi: {filename}")
        st.info(f"📂 Dosya yolu: {os.path.abspath(filename)}")
        
        return filename
    
    def render_labeling_interface(self):
        """Etiketleme arayüzünü render et"""
        
        df = st.session_state.df
        current_idx = st.session_state.current_index
        total_messages = len(df)
        
        if current_idx >= total_messages:
            st.success("🎉 Tüm mesajlar etiketlendi!")
            self.render_completion_interface()
            return
        
        current_row = df.iloc[current_idx]
        
        # Üst bilgi çubuğu
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"Mesaj {current_idx + 1} / {total_messages}")
        
        with col2:
            if st.button("⬅️ Önceki", disabled=current_idx == 0):
                st.session_state.current_index = max(0, current_idx - 1)
                st.rerun()
        
        with col3:
            if st.button("➡️ Sonraki", disabled=current_idx >= total_messages - 1):
                st.session_state.current_index = min(total_messages - 1, current_idx + 1)
                st.rerun()
        
        st.markdown("---")
        
        # Mesaj bilgileri
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Mesaj İçeriği")
            
            # Mesaj kutusu
            st.text_area(
                "Mesaj:",
                current_row['message'],
                height=120,
                disabled=True,
                key=f"message_{current_idx}"
            )
            
            # Ek bilgiler
            info_col1, info_col2 = st.columns(2)
            with info_col1:
                st.write(f"**Gönderen:** {current_row.get('sender', 'N/A')}")
                st.write(f"**Kullanıcı Tipi:** {current_row.get('user_type', 'N/A')}")
            
            with info_col2:
                st.write(f"**Tarih:** {current_row.get('timestamp', 'N/A')}")
                st.write(f"**ID:** {current_row.get('message_id', current_idx + 1)}")
        
        with col2:
            st.subheader("🤖 LLM Tahminleri")
            
            # LLM tahminlerini renkli kutularda göster
            self.render_prediction_box("Sentiment", current_row.get('llm_sentiment', 'N/A'), "🎭")
            self.render_prediction_box("Konu", current_row.get('llm_topic', 'N/A'), "📂")
            self.render_prediction_box("Bot Yanıtı", current_row.get('llm_bot_response', 'N/A'), "🤖")
        
        st.markdown("---")
        
        # Manuel etiketleme bölümü
        st.subheader("✋ Manuel Etiketleme")
        
        # Mevcut etiketleri al (varsa)
        existing_labels = st.session_state.manual_labels.get(current_idx, {})
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            manual_sentiment = st.selectbox(
                "🎭 Doğru Sentiment:",
                self.sentiment_options,
                index=self._get_default_index(
                    self.sentiment_options,
                    existing_labels.get('manual_sentiment', current_row.get('llm_sentiment', 'Nötr'))
                ),
                key=f"sentiment_{current_idx}"
            )
        
        with col2:
            manual_topic = st.selectbox(
                "📂 Doğru Konu:",
                self.dugum_buketi_categories,
                index=self._get_default_index(
                    self.dugum_buketi_categories,
                    existing_labels.get('manual_topic', current_row.get('llm_topic', 'Genel Bilgi'))
                ),
                key=f"topic_{current_idx}"
            )
        
        with col3:
            manual_bot_response = st.selectbox(
                "🤖 Bot Yanıt Verdi mi?:",
                self.bot_response_options,
                index=self._get_default_index(
                    self.bot_response_options,
                    existing_labels.get('manual_bot_response', current_row.get('llm_bot_response', 'Hayır'))
                ),
                key=f"bot_response_{current_idx}"
            )
        
        # Doğruluk göstergeleri
        st.markdown("### 🎯 Doğruluk Durumu")
        acc_col1, acc_col2, acc_col3 = st.columns(3)
        
        with acc_col1:
            sentiment_correct = manual_sentiment == current_row.get('llm_sentiment', '')
            st.write(f"Sentiment: {'✅' if sentiment_correct else '❌'}")
        
        with acc_col2:
            topic_correct = manual_topic == current_row.get('llm_topic', '')
            st.write(f"Konu: {'✅' if topic_correct else '❌'}")
        
        with acc_col3:
            bot_correct = manual_bot_response == current_row.get('llm_bot_response', '')
            st.write(f"Bot Yanıt: {'✅' if bot_correct else '❌'}")
        
        # Kaydetme butonları
        st.markdown("---")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("💾 Kaydet", type="primary"):
                self._save_current_labels(current_idx, manual_sentiment, manual_topic, manual_bot_response)
                st.success("Kaydedildi!")
                st.rerun()
        
        with col2:
            if st.button("💾➡️ Kaydet ve Devam"):
                self._save_current_labels(current_idx, manual_sentiment, manual_topic, manual_bot_response)
                if current_idx < total_messages - 1:
                    st.session_state.current_index += 1
                st.rerun()
        
        with col3:
            if st.button("⏭️ Atla"):
                if current_idx < total_messages - 1:
                    st.session_state.current_index += 1
                st.rerun()
        
        with col4:
            if st.button("🔄 LLM ile Aynı"):
                self._save_current_labels(
                    current_idx,
                    current_row.get('llm_sentiment', 'Nötr'),
                    current_row.get('llm_topic', 'Genel Bilgi'),
                    current_row.get('llm_bot_response', 'Hayır')
                )
                st.success("LLM etiketleri kabul edildi!")
                st.rerun()
        
        # Yorum ekleme
        comment = st.text_input(
            "💬 Yorum (isteğe bağlı):",
            value=existing_labels.get('comment', ''),
            key=f"comment_{current_idx}"
        )
        
        if comment:
            if current_idx not in st.session_state.manual_labels:
                st.session_state.manual_labels[current_idx] = {}
            st.session_state.manual_labels[current_idx]['comment'] = comment
    
    def render_prediction_box(self, label: str, value: str, icon: str):
        """Tahmin kutusunu render et"""
        st.markdown(f"""
        <div style="
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 5px;
            margin: 5px 0;
            border-left: 4px solid #1f77b4;
            color: #000000;
        ">
            <strong style="color: #000000;">{icon} {label}:</strong><br>
            <span style="color: #000000; font-weight: bold;">{value}</span>
        </div>
        """, unsafe_allow_html=True)
    
    def _get_default_index(self, options: list, value: str) -> int:
        """Varsayılan index'i al"""
        try:
            return options.index(value)
        except ValueError:
            return 0
    
    def _save_current_labels(self, index: int, sentiment: str, topic: str, bot_response: str):
        """Mevcut etiketleri kaydet"""
        st.session_state.manual_labels[index] = {
            'manual_sentiment': sentiment,
            'manual_topic': topic,
            'manual_bot_response': bot_response,
            'labeled_at': datetime.now().isoformat()
        }
    
    def render_completion_interface(self):
        """Tamamlama arayüzü"""
        st.balloons()
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Doğruluk Raporu Oluştur", type="primary"):
                self.generate_accuracy_report()
        
        with col2:
            if st.button("💾 Sonuçları İndir"):
                self.download_results()
    
    def generate_accuracy_report(self):
        """Doğruluk raporu oluştur"""
        df = st.session_state.df.copy()
        
        # Manuel etiketleri ekle
        for idx, labels in st.session_state.manual_labels.items():
            if idx < len(df):
                df.loc[idx, 'manual_sentiment'] = labels.get('manual_sentiment')
                df.loc[idx, 'manual_topic'] = labels.get('manual_topic')
                df.loc[idx, 'manual_bot_response'] = labels.get('manual_bot_response')
        
        # Doğruluk hesapla
        sentiment_correct = (df['manual_sentiment'] == df['llm_sentiment']).sum()
        topic_correct = (df['manual_topic'] == df['llm_topic']).sum()
        bot_correct = (df['manual_bot_response'] == df['llm_bot_response']).sum()
        
        total_labeled = len(st.session_state.manual_labels)
        
        # Rapor tablosu
        report_data = {
            'Başlık': ['Sentiment', 'Konu', 'Bot Yanıtı'],
            'Doğru Sayısı': [sentiment_correct, topic_correct, bot_correct],
            'Toplam': [total_labeled, total_labeled, total_labeled],
            'Doğruluk (%)': [
                round((sentiment_correct / total_labeled) * 100, 1) if total_labeled > 0 else 0,
                round((topic_correct / total_labeled) * 100, 1) if total_labeled > 0 else 0,
                round((bot_correct / total_labeled) * 100, 1) if total_labeled > 0 else 0
            ]
        }
        
        report_df = pd.DataFrame(report_data)
        
        st.subheader("📊 Doğruluk Raporu")
        st.dataframe(report_df, use_container_width=True)
        
        # Grafik
        fig = px.bar(
            report_df,
            x='Başlık',
            y='Doğruluk (%)',
            title='LLM Doğruluk Oranları',
            color='Doğruluk (%)',
            color_continuous_scale='RdYlGn'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def download_results(self):
        """Sonuçları indirme"""
        df = st.session_state.df.copy()
        
        # Manuel etiketleri ekle
        for idx, labels in st.session_state.manual_labels.items():
            if idx < len(df):
                for key, value in labels.items():
                    df.loc[idx, key] = value
        
        # Doğruluk sütunları ekle
        df['sentiment_correct'] = df['manual_sentiment'] == df['llm_sentiment']
        df['topic_correct'] = df['manual_topic'] == df['llm_topic']
        df['bot_response_correct'] = df['manual_bot_response'] == df['llm_bot_response']
        
        # CSV olarak indir
        csv = df.to_csv(index=False, encoding='utf-8-sig')
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"manuel_etiketli_veri_{timestamp}.csv"
        
        st.download_button(
            label="📥 CSV İndir",
            data=csv,
            file_name=filename,
            mime='text/csv'
        )

def main():
    """Ana fonksiyon"""
    labeling_system = EnhancedManualLabeling()
    labeling_system.create_enhanced_interface()

if __name__ == "__main__":
    main()