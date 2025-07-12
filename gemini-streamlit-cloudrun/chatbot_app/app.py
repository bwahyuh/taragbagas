# pylint: disable=broad-exception-caught,broad-exception-raised,invalid-name
"""
This module demonstrates the usage of the Gemini API in Vertex AI within a Streamlit application.
"""
import os

from google import genai
import google.auth
from google.genai.types import GenerateContentConfig, Part, ThinkingConfig, EmbedContentConfig
import httpx
import streamlit as st
import vertexai
from vertexai.generative_models import GenerativeModel, Part
# --- Konfigurasi Global Aplikasi ---
PROJECT_ID = "taragbagas-465109"  # Ganti dengan Project ID Anda
LOCATION = "us-central1"          # Ganti dengan region Anda, misal: asia-southeast1
# Kita tidak lagi butuh GenerateContentConfig atau ThinkingConfig di sini
import psycopg2
import pandas as pd
import textwrap # Untuk memotong teks
import time

# --- Library untuk Model SigLIP ---
from transformers import AutoProcessor, AutoModel
import torch
from PIL import Image

# 2. Minta kredensial secara eksplisit dengan cakupan (scope) yang benar
try:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
except Exception as e:
    st.error(f"Gagal melakukan inisialisasi awal Vertex AI: {e}")
    st.stop()

# 3. Inisialisasi Vertex AI dengan kredensial eksplisit tersebut
vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)

# Koneksi dibuat sekali, bisa dipakai semua tab
@st.cache_resource
def get_db_connection():
    """Gets a database connection and caches it for the session."""
    try:
        conn = psycopg2.connect(
            host="34.50.104.207",
            port="5432",
            database="postgres",
            user="postgres",
            password="bagaswahyu"
        )
        return conn
    except psycopg2.OperationalError as e:
        st.error(f"🚨 Kesalahan Koneksi Database: Tidak dapat terhubung ke PostgreSQL. Pastikan database aktif dan kredensial sudah benar. Detail: {e}")
        st.stop()

def _project_id() -> str:
    """Use the Google Auth helper (via the metadata service) to get the Google Cloud Project"""
    try:
        _, project = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError as e:
        raise Exception("Could not automatically determine credentials") from e
    if not project:
        raise Exception("Could not determine project from credentials.")
    return project


def _region() -> str:
    """Use the local metadata service to get the region"""
    try:
        resp = httpx.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/region",
            headers={"Metadata-Flavor": "Google"},
        )
        return resp.text.split("/")[-1]
    except Exception:
        return "us-central1"


MODELS = {
    "gemini-2.0-flash": "Gemini 2.0 Flash",
    "gemini-2.0-flash-lite": "Gemini 2.0 Flash-Lite",
    "gemini-2.5-flash-lite-preview-06-17": "Gemini 2.5 Flash-Lite",
    "gemini-2.5-pro": "Gemini 2.5 Pro",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "model-optimizer-exp-04-09": "Model Optimizer",
}

THINKING_BUDGET_MODELS = {
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite-preview-06-17",
}


@st.cache_resource
def load_client() -> genai.Client:
    """Load Google Gen AI Client."""
    API_KEY = os.environ.get("GOOGLE_API_KEY")
    PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", _project_id())
    LOCATION = os.environ.get("GOOGLE_CLOUD_REGION", _region())

    if not API_KEY and not PROJECT_ID:
        st.error(
            "🚨 Configuration Error: Please set either `GOOGLE_API_KEY` or ensure "
            "Application Default Credentials (ADC) with a Project ID are configured."
        )
        st.stop()
    if not LOCATION:
        st.warning(
            "⚠️ Could not determine Google Cloud Region. Using 'global'. "
            "Ensure GOOGLE_CLOUD_REGION environment variable is set or metadata service is accessible if needed."
        )
        LOCATION = "global"

    return genai.Client(
        vertexai=True,
        project=PROJECT_ID,
        location=LOCATION,
        api_key=API_KEY,
    )


def get_model_name(name: str | None) -> str:
    """Get the formatted model name."""
    if not name:
        return "Gemini"
    return MODELS.get(name, "Gemini")


# --- Helper Functions (Moved outside tabs for reusability) ---

# 1. Definisikan fungsi dialog
@st.dialog("Detail Produk", width="large")
def show_product_details(shoe_data):
    """Menampilkan detail lengkap produk dan tombol order fiktif."""
    st.markdown(f"### {shoe_data['title']}")
    st.image(shoe_data['image_path'], use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Merek:**")
        st.markdown(f"**_{shoe_data['brand']}_**")
    with col2:
        st.markdown(f"**Harga:**")
        st.markdown(f"**_{shoe_data['price']}_**")

    st.markdown("---")

    with st.expander("Lihat Detail Produk"):
        st.write(shoe_data['product_details_clean'])
    
    with st.expander("Fitur"):
        st.write(shoe_data['features_clean'])
    
    st.markdown("---")
    
    order_col, close_col = st.columns([1, 1])
    with order_col:
        if st.button("🛒 Order Sekarang", type="primary", use_container_width=True, key=f"order_{shoe_data['title']}"):
            st.toast(f"🎉 Pesanan untuk '{shoe_data['title']}' telah diterima!", icon="✅")
            time.sleep(2)
            st.rerun()
    with close_col:
        if st.button("Tutup", use_container_width=True, key=f"close_{shoe_data['title']}"):
            st.rerun()

# 2. Fungsi untuk menampilkan grid produk
def display_product_grid(df, key_prefix=""):
    """Menerima DataFrame dan menampilkannya dalam format grid 5 kolom."""
    if df.empty:
        return

    n_cols = 5
    n_rows = (len(df) + n_cols - 1) // n_cols

    for i in range(n_rows):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(df):
                shoe = df.iloc[idx]
                with cols[j]:
                    truncated_title = textwrap.shorten(shoe['title'], width=40, placeholder="...")
                    st.markdown(f"**{truncated_title}**")
                    st.markdown(f"_{shoe['price']}_")
                    st.image(shoe['image_path'], width=150)
                    
                    if st.button("Lihat Detail", key=f"{key_prefix}_btn_{shoe['title']}_{idx}"):
                        show_product_details(shoe.to_dict())
                    
                    st.markdown("---")

# PERBAIKAN: Menambahkan fungsi yang hilang
@st.cache_data
def get_random_products(_conn):
    """Fetches 20 random products to display on the homepage and caches the result."""
    try:
        random_query = "SELECT * FROM products WHERE image_path IS NOT NULL ORDER BY RANDOM() LIMIT 20"
        return pd.read_sql(random_query, _conn)
    except Exception as e:
        st.warning(f"Tidak dapat memuat produk acak. Detail: {e}")
        return pd.DataFrame()


st.link_button(
    "View on GitHub",
    "https://github.com/bwahyuh/taragbagas",
)

cloud_run_service = os.environ.get("K_SERVICE")
if cloud_run_service:
    st.link_button(
        "Open in Cloud Run",
        f"https://console.cloud.google.com/run/detail/us-central1/{cloud_run_service}/source",
    )

# --- LOGIKA MODEL SIGLIP ---
@st.cache_resource
def load_siglip_model():
    """Memuat model dan prosesor SigLIP dan menyimpannya di cache."""
    model_name = "google/siglip-so400m-patch14-384"
    # Menggunakan float32 untuk kompatibilitas CPU di Cloud Run
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

# --- Tampilan Utama Aplikasi ---
st.header("👟 Shoe Recommendation System", divider="rainbow")
client = load_client()

# Memuat koneksi dan model saat aplikasi dimulai
conn = get_db_connection()
siglip_model, siglip_processor = load_siglip_model()
# PERBAIKAN: Menambahkan koma yang hilang untuk memperbaiki ValueError
tab_list = [
    "🛒 Recsys (Keyword)",
    "🗣️ Recsys (Semantic Text)",
    "📸 Recsys (Image)",
    "📸 + ✍️ Recsys (Multimodal)",
    "🤖 RAG Chat"
]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

# --- TAB 1: KEYWORD SEARCH ---
with tab1:
    st.subheader("Pencarian Berdasarkan Nama Produk")
    st.info("💡 Masukkan sebagian nama sepatu (contoh: 'running shoes') untuk mencari rekomendasi.")
    
    selected_shoe_name = st.text_input("Rekomendasikan sepatu yang mirip dengan:", key="keyword_search")
    recommend_button = st.button("Recommend Me!", key="keyword_recommend_button")

    if 'keyword_recommendations' not in st.session_state:
        st.session_state.keyword_recommendations = None
    if 'keyword_matched_title' not in st.session_state:
        st.session_state.keyword_matched_title = None

    if recommend_button and selected_shoe_name:
        with st.spinner(f"Mencari produk yang cocok dengan '{selected_shoe_name}'..."):
            try:
                cur = conn.cursor()
                emb_query = "SELECT text_embedding, title FROM products WHERE LOWER(title) LIKE %s LIMIT 1"
                search_term = f"%{selected_shoe_name.lower()}%"
                cur.execute(emb_query, (search_term,))
                result = cur.fetchone()

                if not result:
                    st.warning("Tidak ada sepatu yang cocok dengan nama tersebut di database.")
                    st.session_state.keyword_recommendations = pd.DataFrame()
                else:
                    shoe_embedding, matched_title = result
                    st.session_state.keyword_matched_title = matched_title
                    st.info(f"Ditemukan: '{matched_title}'. Mencari rekomendasi yang mirip...")
                    rec_query = "SELECT * FROM products WHERE text_embedding IS NOT NULL AND title != %s ORDER BY text_embedding <=> %s ASC LIMIT 20"
                    df_shoes = pd.read_sql(rec_query, conn, params=(matched_title, shoe_embedding))
                    st.session_state.keyword_recommendations = df_shoes
            except Exception as e:
                st.error(f"Terjadi kesalahan saat mengambil data: {e}")
                st.session_state.keyword_recommendations = None
    elif recommend_button and not selected_shoe_name:
        st.session_state.keyword_recommendations = None

    if st.session_state.get('keyword_recommendations') is not None:
        st.success(f"Menampilkan rekomendasi teratas untuk '{st.session_state.keyword_matched_title}':")
        display_product_grid(st.session_state.keyword_recommendations, key_prefix="keyword")
    else:
        st.subheader("Jelajahi Produk Kami")
        random_products = get_random_products(conn)
        display_product_grid(random_products, key_prefix="keyword_random")

# --- TAB 2: SEMANTIC SEARCH ---
with tab2:
    st.subheader("Pencarian Berdasarkan Deskripsi (Teks)")
    st.info("💡 Jelaskan produk yang Anda cari, contoh: 'sandal nyaman untuk traveling musim panas'.")
    
    semantic_search_term = st.text_area("Deskripsikan produk yang Anda cari:", key="semantic_search", height=100)
    semantic_recommend_button = st.button("Recommend Me!", key="semantic_recommend_button")

    if 'semantic_recommendations' not in st.session_state:
        st.session_state.semantic_recommendations = None

    if semantic_recommend_button and semantic_search_term:
        with st.spinner("Menganalisis deskripsi Anda dan mencari produk..."):
            try:
                inputs = siglip_processor(text=[semantic_search_term], return_tensors="pt", padding="max_length")
                with torch.no_grad():
                    input_vector = siglip_model.get_text_features(**inputs).squeeze(0).tolist()
                
                if input_vector:
                    vector_string = str(input_vector)
                    rec_query = "SELECT * FROM products WHERE text_embedding IS NOT NULL ORDER BY text_embedding <=> %s::vector ASC LIMIT 20"
                    df_shoes = pd.read_sql(rec_query, conn, params=(vector_string,))
                    st.session_state.semantic_recommendations = df_shoes
                else:
                    st.warning("Tidak dapat membuat pemahaman dari deskripsi Anda.")
                    st.session_state.semantic_recommendations = pd.DataFrame()
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.session_state.semantic_recommendations = pd.DataFrame()
    elif semantic_recommend_button and not semantic_search_term:
        st.session_state.semantic_recommendations = None

    if st.session_state.get('semantic_recommendations') is not None:
        st.success("Menampilkan rekomendasi berdasarkan deskripsi Anda:")
        display_product_grid(st.session_state.semantic_recommendations, key_prefix="semantic")
    else:
        st.subheader("Jelajahi Produk Kami")
        random_products = get_random_products(conn)
        display_product_grid(random_products, key_prefix="semantic_random")

# --- TAB 3: MULTIMODAL SEARCH ---
with tab3:
    st.subheader("Pencarian Berdasarkan Gambar")
    st.info("💡 Unggah gambar produk untuk menemukan produk yang mirip secara visual.")
    
    uploaded_image = st.file_uploader("Unggah gambar produk...", type=["jpg", "jpeg", "png"], key="image_upload")
    multimodal_recommend_button = st.button("Recommend Me!", key="multimodal_recommend_button")

    if 'multimodal_recommendations' not in st.session_state:
        st.session_state.multimodal_recommendations = None

    if multimodal_recommend_button and uploaded_image is not None:
        with st.spinner("Menganalisis gambar dan mencari produk serupa..."):
            try:
                image = Image.open(uploaded_image).convert("RGB")
                inputs = siglip_processor(images=[image], return_tensors="pt")
                with torch.no_grad():
                    input_vector = siglip_model.get_image_features(**inputs).squeeze(0).tolist()

                if input_vector:
                    vector_string = str(input_vector)
                    rec_query = "SELECT * FROM products WHERE image_embedding IS NOT NULL ORDER BY image_embedding <=> %s::vector ASC LIMIT 20"
                    df_shoes = pd.read_sql(rec_query, conn, params=(vector_string,))
                    st.session_state.multimodal_recommendations = df_shoes
                else:
                    st.warning("Tidak dapat membuat pemahaman dari gambar Anda.")
                    st.session_state.multimodal_recommendations = pd.DataFrame()
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.session_state.multimodal_recommendations = pd.DataFrame()
    elif multimodal_recommend_button and uploaded_image is None:
        st.warning("Silakan unggah gambar terlebih dahulu.")
        st.session_state.multimodal_recommendations = None

    if st.session_state.get('multimodal_recommendations') is not None:
        st.success("Menampilkan rekomendasi berdasarkan gambar Anda:")
        display_product_grid(st.session_state.multimodal_recommendations, key_prefix="multimodal")
    else:
        st.subheader("Jelajahi Produk Kami")
        random_products = get_random_products(conn)
        display_product_grid(random_products, key_prefix="multimodal_random")

# --- TAB 4: MULTIMODAL SEARCH  (Text - Image)---
with tab4:
    st.subheader("Pencarian Multimodal (Teks + Gambar)")
    st.info("💡 Masukkan deskripsi produk dan/atau unggah gambar untuk mencari produk yang paling mirip.")

    # Input gambar di atas input teks
    multi_image = st.file_uploader(
        "Unggah gambar produk:",
        type=["jpg", "jpeg", "png"],
        key="multi_image_upload_vertical"
    )
    if multi_image:
        st.image(multi_image, caption="Preview gambar", width=220)
    
    multi_text = st.text_area(
        "Deskripsikan produk:",
        key="multi_text_input_vertical",
        height=100
    )

    # Tombol di tengah bawah input
    col_btn1, col_btn2, col_btn3 = st.columns([2, 3, 2])
    with col_btn2:
        multi_search_button = st.button(
            "Recommend Me!",
            key="multi_search_button_vertical",
            use_container_width=True
        )

    if 'multimodal_mix_recommendations' not in st.session_state:
        st.session_state.multimodal_mix_recommendations = None

    if multi_search_button and (multi_text or multi_image is not None):
        with st.spinner("Memproses pencarian multimodal..."):
            try:
                text_vector, image_vector = None, None
                rec_query, query_vector = None, None

                # Proses embedding teks jika ada input teks
                if multi_text:
                    text_inputs = siglip_processor(
                        text=[multi_text], return_tensors="pt", padding="max_length"
                    )
                    with torch.no_grad():
                        text_vector = siglip_model.get_text_features(**text_inputs).squeeze(0)
                # Proses embedding gambar jika ada input gambar
                if multi_image is not None:
                    img = Image.open(multi_image).convert("RGB")
                    img_inputs = siglip_processor(images=[img], return_tensors="pt")
                    with torch.no_grad():
                        image_vector = siglip_model.get_image_features(**img_inputs).squeeze(0)
                
                # --- SOLUSI: Logika Pencarian yang Benar ---
                if text_vector is not None and image_vector is not None:
                    # KASUS 1: Teks dan Gambar ada -> Rata-ratakan vektor, cari di embedding gambar
                    query_vector = (text_vector + image_vector) / 2
                    rec_query = "SELECT * FROM products WHERE image_embedding IS NOT NULL ORDER BY image_embedding <=> %s::vector ASC LIMIT 20"
                elif text_vector is not None:
                    # KASUS 2: Hanya Teks -> Cari di embedding teks
                    query_vector = text_vector
                    rec_query = "SELECT * FROM products WHERE text_embedding IS NOT NULL ORDER BY text_embedding <=> %s::vector ASC LIMIT 20"
                elif image_vector is not None:
                    # KASUS 3: Hanya Gambar -> Cari di embedding gambar
                    query_vector = image_vector
                    rec_query = "SELECT * FROM products WHERE image_embedding IS NOT NULL ORDER BY image_embedding <=> %s::vector ASC LIMIT 20"
                
                if query_vector is not None:
                    vector_string = str(query_vector.tolist())
                    df_recs = pd.read_sql(rec_query, conn, params=(vector_string,))
                    st.session_state.multimodal_mix_recommendations = df_recs
                else:
                    st.warning("Minimal satu input (teks/gambar) diperlukan!")
                    st.session_state.multimodal_mix_recommendations = pd.DataFrame()

            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
                st.session_state.multimodal_mix_recommendations = pd.DataFrame()
    elif multi_search_button and (not multi_text and multi_image is None):
        st.warning("Isi teks, upload gambar, atau keduanya!")
        st.session_state.multimodal_mix_recommendations = None

    if st.session_state.get('multimodal_mix_recommendations') is not None:
        st.success("Menampilkan rekomendasi hasil pencarian multimodal Anda:")
        display_product_grid(st.session_state.multimodal_mix_recommendations, key_prefix="multimodal_mix")
    else:
        st.subheader("Jelajahi Produk Kami")
        random_products = get_random_products(conn)
        display_product_grid(random_products, key_prefix="multimodal_mix_random")

# --- TAB 5: RAG-BASED CHAT APPLICATION (MULTIMODAL FINAL) ---
with tab5:
    # --- Main Application UI ---
    st.subheader("🤖 Chat Asisten Belanja Cerdas")
    selected_model = st.selectbox(
        "Select Model:",
        MODELS.keys(),
        format_func=get_model_name,
        key="rag_selected_model" # Menggunakan key unik untuk tab ini
    )

    thinking_budget = None
    if selected_model in THINKING_BUDGET_MODELS:
        thinking_budget_mode = st.selectbox(
            "Thinking budget",
            ("Auto", "Manual", "Off"),
            key="rag_thinking_budget_mode_selectbox", # Key unik
        )

        if thinking_budget_mode == "Manual":
            thinking_budget = st.slider(
                "Thinking budget token limit",
                min_value=0,
                max_value=24576,
                step=1,
                key="rag_thinking_budget_manual_slider", # Key unik
            )
        elif thinking_budget_mode == "Off":
            thinking_budget = 0

    thinking_config = (
        ThinkingConfig(thinking_budget=thinking_budget)
        if thinking_budget is not None
        else None
    )
    st.info(f"Anda bisa bertanya menggunakan teks atau mengunggah gambar produk untuk menemukan barang serupa! Model yang aktif saat ini adalah **{selected_model}**.")
    
    # --- Chat History Management ---
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait produk sepatu kami? Anda bisa bertanya atau unggah gambar."}]
    
    # 1. Tampilkan seluruh history percakapan
    for message_idx, message in enumerate(st.session_state.rag_messages):
        content = message["content"]
        with st.chat_message(message["role"]):
            if isinstance(content, str):
                st.markdown(content)
            elif isinstance(content, dict):
                # Tampilan untuk pesan PENGGUNA (mungkin ada gambar)
                if "text" in content:
                    if content.get("image_bytes"):
                        st.image(content["image_bytes"], width=200, caption="Gambar yang Anda unggah")
                    st.markdown(content["text"])
                # Tampilan untuk respons ASISTEN (reasoning + grid produk)
                elif "reasoning" in content:
                    st.markdown(content["reasoning"])
                    if "recommendations_df" in content and not content["recommendations_df"].empty:
                        # Buat key unik untuk setiap grid produk dalam history
                        unique_key_prefix = f"rag_multi_{message_idx}"
                        display_product_grid(content["recommendations_df"], key_prefix=unique_key_prefix)
                    else:
                        st.write("Tidak ada produk spesifik yang cocok untuk direkomendasikan.")

    # 2. Area Input Pengguna
    uploaded_image = st.file_uploader("Unggah gambar untuk mencari produk serupa", type=["png", "jpg", "jpeg"], key="rag_file_uploader")
    
    if prompt := st.chat_input("Tulis pertanyaan atau deskripsikan gambar..."):
        user_message_content = {"text": prompt}
        if uploaded_image is not None:
            user_message_content["image_bytes"] = uploaded_image.getvalue()
        
        st.session_state.rag_messages.append({"role": "user", "content": user_message_content})
        st.rerun()

    # 3. Proses pesan terakhir dari user jika belum diproses
    last_message = st.session_state.rag_messages[-1]
    if last_message["role"] == "user" and not last_message.get("processed", False):
        
        with st.chat_message("assistant"):
            with st.spinner("Mencari & menganalisis produk..."):
                
                user_content = last_message["content"]
                prompt_text = user_content["text"]
                image_bytes = user_content.get("image_bytes")
                
                context_df = pd.DataFrame()
                context_for_llm = "[]" # Default ke JSON array kosong
                # Normalisasi input pengguna: huruf kecil dan apostrof standar
                normalized_prompt_text = prompt_text.lower().replace("’", "'")

                try:
                    # --- Tahap 1: Retrieval (Pengambilan Kandidat) ---
                    text_vector, image_vector = None, None
                    
                    if prompt_text:
                        text_inputs = siglip_processor(text=[prompt_text], return_tensors="pt", padding="max_length")
                        with torch.no_grad():
                            text_vector = siglip_model.get_text_features(**text_inputs).squeeze(0)

                    if image_bytes:
                        from PIL import Image
                        import io
                        input_image = Image.open(io.BytesIO(image_bytes))
                        image_inputs = siglip_processor(images=[input_image], return_tensors="pt")
                        with torch.no_grad():
                            image_vector = siglip_model.get_image_features(**image_inputs).squeeze(0)

                    query_vector, search_query, search_type_info = None, None, "Input tidak valid."

                    if text_vector is not None and image_vector is not None:
                        st.info("🔎 Mencari berdasarkan kombinasi teks dan gambar...")
                        query_vector = (text_vector + image_vector) / 2
                        search_query = "SELECT * FROM products ORDER BY image_embedding <=> %s::vector LIMIT 10;"
                        search_type_info = "berdasarkan kombinasi teks dan gambar yang Anda berikan"
                    elif image_vector is not None:
                        st.info("🔎 Mencari berdasarkan kemiripan gambar...")
                        query_vector = image_vector
                        search_query = "SELECT * FROM products ORDER BY image_embedding <=> %s::vector LIMIT 10;"
                        search_type_info = "berdasarkan gambar yang diunggah pengguna"
                    elif text_vector is not None:
                        st.info("🔎 Mencari berdasarkan deskripsi teks...")
                        query_vector = text_vector
                        search_query = "SELECT * FROM products ORDER BY text_embedding <=> %s::vector LIMIT 10;"
                        search_type_info = "berdasarkan deskripsi teks pengguna"

                    if query_vector is not None and search_query:
                        vector_string = str(query_vector.tolist())
                        # Menambahkan kolom 'features_clean' ke query
                        context_df = pd.read_sql(search_query, conn, params=(vector_string,))
                        
                        # --- Tahap 2: Persiapan Konteks untuk LLM ---
                        if not context_df.empty:
                            # Buat salinan untuk normalisasi agar data asli tetap utuh
                            temp_df = context_df.copy()
                            # Normalisasi huruf menjadi kecil dan ganti apostrof untuk perbandingan oleh LLM
                            temp_df['title'] = temp_df['title'].str.lower().str.replace("’", "'", regex=False)
                            temp_df['brand'] = temp_df['brand'].str.lower().str.replace("’", "'", regex=False)
                            # Normalisasi kolom fitur juga
                            if 'features_clean' in temp_df.columns:
                                temp_df['features_clean'] = temp_df['features_clean'].astype(str).str.lower().str.replace("’", "'", regex=False)

                            # Sertakan 'features_clean' dalam data yang dikirim ke LLM
                            context_for_llm = temp_df[['title', 'brand', 'price', 'features_clean']].to_json(orient='records', indent=2)
                        else:
                            context_for_llm = "[]" # JSON array kosong jika tidak ada hasil
                    else:
                        context_for_llm = "[]" # JSON array kosong jika input tidak memadai

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat mengambil data dari database: {e}")
                    context_for_llm = "[]" # JSON array kosong jika terjadi error

                # --- Tahap 3: Generation (Memanggil Gemini dengan Prompt yang Lebih Detail) ---
                final_prompt = f"""
                PERINTAH: Anda adalah asisten belanja ahli. Tugas Anda adalah memberikan rekomendasi produk yang detail dan membantu dari KONTEKS berdasarkan PERTANYAAN PENGGUNA. Jawab HANYA dalam format JSON.

                LANGKAH-LANGKAH LOGIS:
                1.  **Analisis & Filter**: Identifikasi kata kunci dari `PERTANYAAN PENGGUNA`. Filter produk dalam `KONTEKS PRODUK` yang 'title' atau 'brand'-nya mengandung kata kunci tersebut.
                2.  **Buat Respons Detail**:
                    - "reasoning": (string dalam format Markdown)
                      - Mulai dengan paragraf pembuka yang ramah dan sebutkan kata kunci yang Anda temukan.
                      - Untuk SETIAP produk yang direkomendasikan, buat daftar berpoin dengan format ini:
                        * **[Judul Produk Lengkap]**:
                          - **Kecocokan Permintaan**: Jelaskan secara singkat mengapa produk ini cocok (misalnya, "Mereknya 'pantofola d'oro' sesuai dengan yang Anda cari.").
                          - **Fitur Utama**: Berdasarkan kolom `features_clean`, tulis deskripsi detail dalam 2-4 kalimat yang merangkum fitur-fitur paling menonjol dan menjelaskan manfaatnya bagi pengguna.
                    - "recommendations": (list of objects) Daftar HANYA `title` dari semua produk yang cocok (dalam huruf kecil). Setiap item HARUS berupa objek JSON dengan satu kunci "title".

                ---
                CONTOH OUTPUT JSON YANG DIHARAPKAN:
                {{
                  "reasoning": "Tentu, saya menemukan beberapa sepatu dari merek 'pantofola d'oro' yang mungkin Anda sukai:\\n\\n* **pantofola d'oro 1886 men's new star soccer shoe**:\\n  - **Kecocokan Permintaan**: Mereknya sesuai dengan yang Anda cari.\\n  - **Fitur Utama**: Sepatu ini dibuat di Italia dan dirancang untuk pemain yang menghargai kualitas. Bagian atasnya terbuat dari 100% kulit anak sapi yang sangat lembut, memberikan sentuhan dan kontrol bola yang luar biasa. Sol luarnya dijahit langsung ke bagian atas untuk meningkatkan daya tahan.\\n* **pantofola d'oro 1886 men's piceno soccer shoe**:\\n  - **Kecocokan Permintaan**: Produk ini juga dari merek 'pantofola d'oro'.\\n  - **Fitur Utama**: Sepatu ini memiliki desain klasik dengan material kulit premium. Ini memberikan kenyamanan optimal dan fleksibilitas saat bergerak. Sol karetnya dirancang untuk memberikan cengkeraman yang andal di berbagai jenis permukaan lapangan.",
                  "recommendations": [
                    {{ "title": "pantofola d'oro 1886 men's new star soccer shoe" }},
                    {{ "title": "pantofola d'oro 1886 men's piceno soccer shoe" }}
                  ]
                }}
                ---
                KONTEKS PRODUK (JSON Array, semua huruf kecil):
                {context_for_llm}
                ---
                PERTANYAAN PENGGUNA (huruf kecil, apostrof standar):
                {normalized_prompt_text}
                ---
                JSON OUTPUT:
                """
                
                try:
                    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
                    rag_model = GenerativeModel(selected_model)
                    rag_config = {
                        "temperature": 0.0, 
                        "max_output_tokens": 8192,
                        "top_p": 0.95
                    }
                    
                    response_text = rag_model.generate_content(final_prompt, generation_config=rag_config).text
                    
                    import json, re
                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if not match: raise ValueError("Tidak ditemukan format JSON yang valid dalam respons AI.")
                    
                    parsed_response = json.loads(match.group(0))
                    reasoning = parsed_response.get("reasoning", "Berikut rekomendasi saya.")
                    
                    # --- SOLUSI: Kode yang lebih fleksibel untuk memproses rekomendasi ---
                    recommendations_list = parsed_response.get("recommendations", [])
                    recommended_titles_lower = []
                    if recommendations_list:
                        # Periksa tipe elemen pertama untuk menentukan cara memproses
                        if isinstance(recommendations_list[0], dict):
                            # Kasus ideal: AI mengembalikan list of dictionaries
                            recommended_titles_lower = [rec.get("title") for rec in recommendations_list if rec.get("title")]
                        elif isinstance(recommendations_list[0], str):
                            # Kasus fallback: AI mengembalikan list of strings
                            recommended_titles_lower = recommendations_list
                    
                    # Ambil data lengkap dari DataFrame asli (sebelum di-lowercase) untuk ditampilkan
                    recommendations_df = pd.DataFrame()
                    if not context_df.empty and recommended_titles_lower:
                        # Normalisasi judul di DataFrame asli untuk perbandingan yang akurat
                        recommendations_df = context_df[context_df['title'].str.lower().str.replace("’", "'", regex=False).isin(recommended_titles_lower)].copy()
                    
                    assistant_response_content = {
                        "reasoning": reasoning,
                        "recommendations_df": recommendations_df
                    }
                    st.session_state.rag_messages[-1]["processed"] = True
                    st.session_state.rag_messages.append({"role": "assistant", "content": assistant_response_content})
                    st.rerun()

                except Exception as e:
                    st.error(f"Gagal memproses respons dari AI: {e}")
                    # Menampilkan prompt dan respons mentah untuk debugging
                    with st.expander("Lihat Detail Debug"):
                        st.code(final_prompt, language="markdown")
                        st.code(response_text, language="text")
