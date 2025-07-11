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
LOCATION = "us-central1"           # Ganti dengan region Anda, misal: asia-southeast1
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

selected_model = st.radio(
    "Select Model:",
    MODELS.keys(),
    format_func=get_model_name,
    key="selected_model",
    horizontal=True,
)

thinking_budget = None
if selected_model in THINKING_BUDGET_MODELS:
    thinking_budget_mode = st.selectbox(
        "Thinking budget",
        ("Auto", "Manual", "Off"),
        key="thinking_budget_mode_selectbox",
    )

    if thinking_budget_mode == "Manual":
        thinking_budget = st.slider(
            "Thinking budget token limit",
            min_value=0,
            max_value=24576,
            step=1,
            key="thinking_budget_manual_slider",
        )
    elif thinking_budget_mode == "Off":
        thinking_budget = 0

thinking_config = (
    ThinkingConfig(thinking_budget=thinking_budget)
    if thinking_budget is not None
    else None
)
# PERBAIKAN: Menambahkan koma yang hilang untuk memperbaiki ValueError
tab_list = [
    "✍️ Freeform",
    "📖 Generate Story",
    "📢 Marketing Campaign",
    "🖼️ Image Playground",
    "🎬 Video Playground",
    "🛒 Recsys (Keyword)",
    "🗣️ Recsys (Semantic Text)",
    "📸 Recsys (Image)",
    "📸 + ✍️ Recsys (Multimodal)",
    "🤖 RAG Chat",
    "playtab10"
]
freeform_tab, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(tab_list)

with freeform_tab:
    st.subheader("Enter Your Own Prompt")

    temperature = st.slider(
        "Select the temperature (Model Randomness):",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.05,
        key="temperature",
    )

    max_output_tokens = st.slider(
        "Maximum Number of Tokens to Generate:",
        min_value=1,
        max_value=8192,
        value=2048,
        step=1,
        key="max_output_tokens",
    )

    top_p = st.slider(
        "Select the Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.95,
        step=0.05,
        key="top_p",
    )

    prompt = st.text_area(
        "Enter your prompt here...",
        key="prompt",
        height=200,
    )

    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        top_p=top_p,
        thinking_config=thinking_config,
    )

    generate_freeform = st.button("Generate", key="generate_freeform")
    if generate_freeform and prompt:
        with st.spinner(
            f"Generating response using {get_model_name(selected_model)} ..."
        ):
            first_tab1, first_tab2 = st.tabs(["Response", "Prompt"])
            with first_tab1:
                response = client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=config,
                ).text

                if response:
                    st.markdown(response)
            with first_tab2:
                st.markdown(
                    f"""Parameters:\n- Model ID: `{selected_model}`\n- Temperature: `{temperature}`\n- Top P: `{top_p}`\n- Max Output Tokens: `{max_output_tokens}`\n"""
                )
                if thinking_budget is not None:
                    st.markdown(f"- Thinking Budget: `{thinking_budget}`\n")
                st.code(prompt, language="markdown")

with tab1:
    st.subheader("Generate a story")

    # Story premise
    character_name = st.text_input(
        "Enter character name: \n\n", key="character_name", value="Mittens"
    )
    character_type = st.text_input(
        "What type of character is it? \n\n", key="character_type", value="Cat"
    )
    character_persona = st.text_input(
        "What personality does the character have? \n\n",
        key="character_persona",
        value="Mittens is a very friendly cat.",
    )
    character_location = st.text_input(
        "Where does the character live? \n\n",
        key="character_location",
        value="Andromeda Galaxy",
    )
    story_premise = st.multiselect(
        "What is the story premise? (can select multiple) \n\n",
        [
            "Love",
            "Adventure",
            "Mystery",
            "Horror",
            "Comedy",
            "Sci-Fi",
            "Fantasy",
            "Thriller",
        ],
        key="story_premise",
        default=["Love", "Adventure"],
    )
    creative_control = st.radio(
        "Select the creativity level: \n\n",
        ["Low", "High"],
        key="creative_control",
        horizontal=True,
    )
    length_of_story = st.radio(
        "Select the length of the story: \n\n",
        ["Short", "Long"],
        key="length_of_story",
        horizontal=True,
    )

    if creative_control == "Low":
        temperature = 0.30
    else:
        temperature = 0.95

    if length_of_story == "Short":
        max_output_tokens = 2048
    else:
        max_output_tokens = 8192

    prompt = f"""Write a {length_of_story} story based on the following premise: \n
  character_name: {character_name} \n
  character_type: {character_type} \n
  character_persona: {character_persona} \n
  character_location: {character_location} \n
  story_premise: {",".join(story_premise)} \n
  If the story is "short", then make sure to have 5 chapters or else if it is "long" then 10 chapters.
  Important point is that each chapters should be generated based on the premise given above.
  First start by giving the book introduction, chapter introductions and then each chapter. It should also have a proper ending.
  The book should have prologue and epilogue.
  """
    config = GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        thinking_config=thinking_config,
    )

    generate_t2t = st.button("Generate my story", key="generate_t2t")
    if generate_t2t and prompt:
        with st.spinner(
            f"Generating your story using {get_model_name(selected_model)} ..."
        ):
            first_tab1, first_tab2 = st.tabs(["Story", "Prompt"])
            with first_tab1:
                response = client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=config,
                ).text

                if response:
                    st.write("Your story:")
                    st.write(response)
            with first_tab2:
                st.markdown(
                    f"""Parameters:\n- Model ID: `{selected_model}`\n- Temperature: `{temperature}`\n- Max Output Tokens: `{max_output_tokens}`\n"""
                )
                if thinking_budget is not None:
                    st.markdown(f"- Thinking Budget: `{thinking_budget}`\n")
                st.code(prompt, language="markdown")

with tab2:
    st.subheader("Generate your marketing campaign")

    product_name = st.text_input(
        "What is the name of the product? \n\n", key="product_name", value="ZomZoo"
    )
    product_category = st.radio(
        "Select your product category: \n\n",
        ["Clothing", "Electronics", "Food", "Health & Beauty", "Home & Garden"],
        key="product_category",
        horizontal=True,
    )
    st.write("Select your target audience: ")
    target_audience_age = st.radio(
        "Target age: \n\n",
        ["18-24", "25-34", "35-44", "45-54", "55-64", "65+"],
        key="target_audience_age",
        horizontal=True,
    )
    target_audience_location = st.radio(
        "Target location: \n\n",
        ["Urban", "Suburban", "Rural"],
        key="target_audience_location",
        horizontal=True,
    )
    st.write("Select your marketing campaign goal: ")
    campaign_goal = st.multiselect(
        "Select your marketing campaign goal: \n\n",
        [
            "Increase brand awareness",
            "Generate leads",
            "Drive sales",
            "Improve brand sentiment",
        ],
        key="campaign_goal",
        default=["Increase brand awareness", "Generate leads"],
    )
    if campaign_goal is None:
        campaign_goal = ["Increase brand awareness", "Generate leads"]
    brand_voice = st.radio(
        "Select your brand voice: \n\n",
        ["Formal", "Informal", "Serious", "Humorous"],
        key="brand_voice",
        horizontal=True,
    )
    estimated_budget = st.radio(
        "Select your estimated budget ($): \n\n",
        ["1,000-5,000", "5,000-10,000", "10,000-20,000", "20,000+"],
        key="estimated_budget",
        horizontal=True,
    )

    prompt = f"""Generate a marketing campaign for {product_name}, a {product_category} designed for the age group: {target_audience_age}.
  The target location is this: {target_audience_location}.
  Aim to primarily achieve {campaign_goal}.
  Emphasize the product's unique selling proposition while using a {brand_voice} tone of voice.
  Allocate the total budget of {estimated_budget}.
  With these inputs, make sure to follow following guidelines and generate the marketing campaign with proper headlines: \n
  - Briefly describe company, its values, mission, and target audience.
  - Highlight any relevant brand guidelines or messaging frameworks.
  - Provide a concise overview of the campaign's objectives and goals.
  - Briefly explain the product or service being promoted.
  - Define your ideal customer with clear demographics, psychographics, and behavioral insights.
  - Understand their needs, wants, motivations, and pain points.
  - Clearly articulate the desired outcomes for the campaign.
  - Use SMART goals (Specific, Measurable, Achievable, Relevant, and Time-bound) for clarity.
  - Define key performance indicators (KPIs) to track progress and success.
  - Specify the primary and secondary goals of the campaign.
  - Examples include brand awareness, lead generation, sales growth, or website traffic.
  - Clearly define what differentiates your product or service from competitors.
  - Emphasize the value proposition and unique benefits offered to the target audience.
  - Define the desired tone and personality of the campaign messaging.
  - Identify the specific channels you will use to reach your target audience.
  - Clearly state the desired action you want the audience to take.
  - Make it specific, compelling, and easy to understand.
  - Identify and analyze your key competitors in the market.
  - Understand their strengths and weaknesses, target audience, and marketing strategies.
  - Develop a differentiation strategy to stand out from the competition.
  - Define how you will track the success of the campaign.
  - Utilize relevant KPIs to measure performance and return on investment (ROI).
  Give proper bullet points and headlines for the marketing campaign. Do not produce any empty lines.
  Be very succinct and to the point.
  """

    config = GenerateContentConfig(
        temperature=0.8,
        max_output_tokens=8192,
        thinking_config=thinking_config,
    )

    generate_t2t = st.button("Generate my campaign", key="generate_campaign")
    if generate_t2t and prompt:
        second_tab1, second_tab2 = st.tabs(["Campaign", "Prompt"])
        with st.spinner(
            f"Generating your marketing campaign using {get_model_name(selected_model)} ..."
        ):
            with second_tab1:
                response = client.models.generate_content(
                    model=selected_model,
                    contents=prompt,
                    config=config,
                ).text
                if response:
                    st.write("Your marketing campaign:")
                    st.write(response)
            with second_tab2:
                st.code(prompt, language="markdown")

with tab3:
    st.subheader("Image Playground")

    furniture, oven, er_diagrams, glasses, math_reasoning = st.tabs(
        [
            "🛋️ Furniture recommendation",
            "🔥 Oven Instructions",
            "📊 ER Diagrams",
            "👓 Glasses",
            "🧮 Math Reasoning",
        ],
    )

    with furniture:
        st.markdown(
            """In this demo, you will be presented with a scene (e.g., a living room) and will use the Gemini model to perform visual understanding. You will see how Gemini can be used to recommend an item (e.g., a chair) from a list of furniture options as input. You can use Gemini to recommend a chair that would complement the given scene and will be provided with its rationale for such selections from the provided list."""
        )

        room_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/rooms/living_room.jpeg"
        chair_1_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/chair1.jpeg"
        chair_2_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/chair2.jpeg"
        chair_3_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/chair3.jpeg"
        chair_4_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/retail-recommendations/furnitures/chair4.jpeg"

        st.image(room_image_uri, width=350, caption="Image of a living room")
        st.image(
            [
                chair_1_image_uri,
                chair_2_image_uri,
                chair_3_image_uri,
                chair_4_image_uri,
            ],
            width=200,
            caption=["Chair 1", "Chair 2", "Chair 3", "Chair 4"],
        )

        st.write(
            "Our expectation: Recommend a chair that would complement the given image of a living room."
        )
        content = [
            "Consider the following chairs:",
            "chair 1:",
            Part.from_uri(uri=chair_1_image_uri, mime_type="image/jpeg"),
            "chair 2:",
            Part.from_uri(uri=chair_2_image_uri, mime_type="image/jpeg"),
            "chair 3:",
            Part.from_uri(uri=chair_3_image_uri, mime_type="image/jpeg"),
            "and",
            "chair 4:",
            Part.from_uri(uri=chair_4_image_uri, mime_type="image/jpeg"),
            "\n"
            "For each chair, explain why it would be suitable or not suitable for the following room:",
            Part.from_uri(uri=room_image_uri, mime_type="image/jpeg"),
            "Only recommend for the room provided and not other rooms. Provide your recommendation in a table format with chair name and reason as columns.",
        ]

        tab1, tab2 = st.tabs(["Response", "Prompt"])
        generate_image_description = st.button(
            "Generate recommendation....", key="generate_image_description"
        )
        with tab1:
            if generate_image_description and content:
                with st.spinner(
                    f"Generating recommendation using {get_model_name(selected_model)} ..."
                ):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=content,
                        config=config,
                    ).text
                    st.markdown(response)
        with tab2:
            st.write("Prompt used:")
            st.code(content, language="markdown")

    with oven:
        stove_screen_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/stove.jpg"
        st.write(
            "Equipped with the ability to extract information from visual elements on screens, Gemini can analyze screenshots, icons, and layouts to provide a holistic understanding of the depicted scene."
        )
        st.image(stove_screen_uri, width=350, caption="Image of a oven")
        st.write(
            "Our expectation: Provide instructions for resetting the clock on this appliance in English"
        )
        prompt = """How can I reset the clock on this appliance? Provide the instructions in English.
If instructions include buttons, also explain where those buttons are physically located.
"""
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        generate_instructions_description = st.button(
            "Generate instructions", key="generate_instructions_description"
        )
        with tab1:
            if generate_instructions_description and prompt:
                with st.spinner(
                    f"Generating instructions using {get_model_name(selected_model)}..."
                ):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=[
                            Part.from_uri(
                                uri=stove_screen_uri, mime_type="image/jpeg"
                            ),
                            prompt,
                        ],
                    ).text
                    st.markdown(response)
        with tab2:
            st.write("Prompt used:")
            st.code(prompt, language="markdown")

    with er_diagrams:
        er_diag_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/er.png"
        st.write(
            "Gemini multimodal capabilities empower it to comprehend diagrams and take actionable steps, such as optimization or code generation. The following example demonstrates how Gemini can decipher an Entity Relationship (ER) diagram."
        )
        st.image(er_diag_uri, width=350, caption="Image of an ER diagram")
        st.write(
            "Our expectation: Document the entities and relationships in this ER diagram."
        )
        prompt = """Document the entities and relationships in this ER diagram.
        """
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        er_diag_img_description = st.button("Generate!", key="er_diag_img_description")
        with tab1:
            if er_diag_img_description and prompt:
                with st.spinner("Generating..."):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=[
                            Part.from_uri(uri=er_diag_uri, mime_type="image/jpeg"),
                            prompt,
                        ],
                    ).text
        with tab2:
            st.write("Prompt used:")
            st.code(prompt, language="markdown")

    with glasses:
        compare_img_1_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/glasses1.jpg"
        compare_img_2_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/glasses2.jpg"

        st.write(
            """Gemini is capable of image comparison and providing recommendations. This can be useful in industries like e-commerce and retail.
          Below is an example of choosing which pair of glasses would be better suited to various face types:"""
        )
        face_type = st.radio(
            "What is your face shape?",
            ["Oval", "Round", "Square", "Heart", "Diamond"],
            key="face_type",
            horizontal=True,
        )
        output_type = st.radio(
            "Select the output type",
            ["text", "table", "json"],
            key="output_type",
            horizontal=True,
        )
        st.image(
            [compare_img_1_uri, compare_img_2_uri],
            width=350,
            caption=["Glasses type 1", "Glasses type 2"],
        )
        st.write(
            f"Our expectation: Suggest which glasses type is better for the {face_type} face shape"
        )
        content = [
            f"""Which of these glasses you recommend for me based on the shape of my face:{face_type}?
      I have an {face_type} shape face.
      Glasses 1: """,
            Part.from_uri(uri=compare_img_1_uri, mime_type="image/jpeg"),
            """
      Glasses 2: """,
            Part.from_uri(uri=compare_img_2_uri, mime_type="image/jpeg"),
            f"""
      Explain how you made to this decision.
      Provide your recommendation based on my face shape, and reasoning for each in {output_type} format.
      """,
        ]
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        compare_img_description = st.button(
            "Generate recommendation!", key="compare_img_description"
        )
        with tab1:
            if compare_img_description and content:
                with st.spinner(
                    f"Generating recommendations using {get_model_name(selected_model)}..."
                ):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=[
                            Part.from_uri(uri=er_diag_uri, mime_type="image/jpeg"),
                            content,
                        ],
                    ).text
                    st.markdown(response)
        with tab2:
            st.write("Prompt used:")
            st.code(content, language="markdown")

    with math_reasoning:
        math_image_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/math_beauty.jpg"

        st.write(
            "Gemini can also recognize math formulas and equations and extract specific information from them. This capability is particularly useful for generating explanations for math problems, as shown below."
        )
        st.image(math_image_uri, width=350, caption="Image of a math equation")
        st.markdown(
            """
        Our expectation: Ask questions about the math equation as follows:
        - Extract the formula.
        - What is the symbol right before Pi? What does it mean?
        - Is this a famous formula? Does it have a name?
          """
        )
        prompt = """
Follow the instructions.
Surround math expressions with $.
Use a table with a row for each instruction and its result.

INSTRUCTIONS:
- Extract the formula.
- What is the symbol right before Pi? What does it mean?
- Is this a famous formula? Does it have a name?
"""
        tab1, tab2 = st.tabs(["Response", "Prompt"])
        math_image_description = st.button(
            "Generate answers!", key="math_image_description"
        )
        with tab1:
            if math_image_description and prompt:
                with st.spinner(
                    f"Generating answers for formula using {get_model_name(selected_model)}..."
                ):
                    response = client.models.generate_content(
                        model=selected_model,
                        contents=[
                            Part.from_uri(
                                uri=math_image_uri, mime_type="image/jpeg"
                            ),
                            prompt,
                        ],
                    ).text
                    st.markdown(response)
                    st.markdown("\n\n\n")
        with tab2:
            st.write("Prompt used:")
            st.code(prompt, language="markdown")

with tab4:
    st.subheader("Video Playground")

    vide_desc, video_tags, video_highlights, video_geolocation = st.tabs(
        [
            "📄 Description",
            "🏷️ Tags",
            "✨ Highlights",
            "📍 Geolocation",
        ]
    )

    with vide_desc:
        st.markdown(
            """Gemini can also provide the description of what is going on in the video:"""
        )
        video_desc_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/mediterraneansea.mp4"

        if video_desc_uri:
            st.video(video_desc_uri)
            st.write("Our expectation: Generate the description of the video")
            prompt = """Describe what is happening in the video and answer the following questions: \n
      - What am I looking at? \n
      - Where should I go to see it? \n
      - What are other top 5 places in the world that look like this?
      """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            vide_desc_description = st.button(
                "Generate video description", key="vide_desc_description"
            )
            with tab1:
                if vide_desc_description and prompt:
                    with st.spinner(
                        f"Generating video description using {get_model_name(selected_model)} ..."
                    ):
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[
                                Part.from_uri(
                                    uri=video_desc_uri, mime_type="video/mp4"
                                ),
                                prompt,
                            ],
                        ).text
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.code(prompt, language="markdown")

    with video_tags:
        st.markdown(
            """Gemini can also extract tags throughout a video, as shown below:."""
        )
        video_tags_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/photography.mp4"

        if video_tags_uri:
            st.video(video_tags_uri)
            st.write("Our expectation: Generate the tags for the video")
            prompt = """Answer the following questions using the video only:
            1. What is in the video?
            2. What objects are in the video?
            3. What is the action in the video?
            4. Provide 5 best tags for this video?
            Give the answer in the table format with question and answer as columns.
      """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            video_tags_description = st.button(
                "Generate video tags", key="video_tags_description"
            )
            with tab1:
                if video_tags_description and prompt:
                    with st.spinner(
                        f"Generating video description using {get_model_name(selected_model)} ..."
                    ):
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[
                                Part.from_uri(
                                    uri=video_tags_uri, mime_type="video/mp4"
                                ),
                                prompt,
                            ],
                        ).text
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.code(prompt, language="markdown")

    with video_highlights:
        st.markdown(
            """Below is another example of using Gemini to ask questions about objects, people or the context, as shown in the video about Pixel 8 below:"""
        )
        video_highlights_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/pixel8.mp4"

        if video_highlights_uri:
            st.video(video_highlights_uri)
            st.write("Our expectation: Generate the highlights for the video")
            prompt = """Answer the following questions using the video only:
What is the profession of the girl in this video?
Which all features of the phone are highlighted here?
Summarize the video in one paragraph.
Provide the answer in table format.
      """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            video_highlights_description = st.button(
                "Generate video highlights", key="video_highlights_description"
            )
            with tab1:
                if video_highlights_description and prompt:
                    with st.spinner(
                        f"Generating video highlights using {get_model_name(selected_model)} ..."
                    ):
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[
                                Part.from_uri(
                                    uri=video_highlights_uri, mime_type="video/mp4"
                                ),
                                prompt,
                            ],
                        ).text
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.code(prompt, language="markdown")

    with video_geolocation:
        st.markdown(
            """Even in short, detail-packed videos, Gemini can identify the locations."""
        )
        video_geolocation_uri = "https://storage.googleapis.com/github-repo/img/gemini/multimodality_usecases_overview/bus.mp4"

        if video_geolocation_uri:
            st.video(video_geolocation_uri)
            st.markdown(
                """Our expectation: \n
      Answer the following questions from the video:
        - What is this video about?
        - How do you know which city it is?
        - What street is this?
        - What is the nearest intersection?
      """
            )
            prompt = """Answer the following questions using the video only:
      What is this video about?
      How do you know which city it is?
      What street is this?
      What is the nearest intersection?
      Answer the following questions in a table format with question and answer as columns.
      """
            tab1, tab2 = st.tabs(["Response", "Prompt"])
            video_geolocation_description = st.button(
                "Generate", key="video_geolocation_description"
            )
            with tab1:
                if video_geolocation_description and prompt:
                    with st.spinner(
                        f"Generating location tags using {get_model_name(selected_model)} ..."
                    ):
                        response = client.models.generate_content(
                            model=selected_model,
                            contents=[
                                Part.from_uri(
                                    uri=video_geolocation_uri,
                                    mime_type="video/mp4",
                                ),
                                prompt,
                            ],
                        ).text
                        st.markdown(response)
                        st.markdown("\n\n\n")
            with tab2:
                st.write("Prompt used:")
                st.code(prompt, language="markdown")


# --- TAB 5: KEYWORD SEARCH ---
with tab5:
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

# --- TAB 6: SEMANTIC SEARCH ---
with tab6:
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

# --- TAB 7: MULTIMODAL SEARCH ---
with tab7:
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

# --- TAB 8: MULTIMODAL SEARCH  (Text - Image)---
with tab8:
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
                text_vector = None
                image_vector = None

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
                # Gabungkan dua embedding jika keduanya ada
                if text_vector is not None and image_vector is not None:
                    combined_vector = (text_vector + image_vector) / 2
                elif text_vector is not None:
                    combined_vector = text_vector
                elif image_vector is not None:
                    combined_vector = image_vector
                else:
                    st.warning("Minimal satu input (teks/gambar) diperlukan!")
                    st.session_state.multimodal_mix_recommendations = pd.DataFrame()
                    combined_vector = None

                if combined_vector is not None:
                    vector_string = str(combined_vector.tolist())
                    rec_query = (
                        "SELECT * FROM products WHERE image_embedding IS NOT NULL "
                        "ORDER BY image_embedding <=> %s::vector ASC LIMIT 20"
                    )
                    df_recs = pd.read_sql(rec_query, conn, params=(vector_string,))
                    st.session_state.multimodal_mix_recommendations = df_recs
                else:
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

# --- TAB 9: RAG-BASED CHAT APPLICATION (MULTIMODAL FINAL) ---
with tab9:
    # --- Main Application ---
    st.subheader("🤖 Chat Asisten Belanja Cerdas")
    st.info("Anda bisa bertanya menggunakan teks atau mengunggah gambar produk untuk menemukan barang serupa!")

    # Inisialisasi & Tampilan Chat History
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [{"role": "assistant", "content": "Halo! Ada yang bisa saya bantu terkait produk sepatu kami? Anda bisa bertanya atau unggah gambar."}]
    
    # 1. Tampilkan seluruh history percakapan
    for message in st.session_state.rag_messages:
        content = message["content"]
        with st.chat_message(message["role"]):
            # Jika konten adalah sapaan awal (string)
            if isinstance(content, str):
                st.markdown(content)
            # Jika konten adalah pesan terstruktur (dictionary)
            elif isinstance(content, dict):
                # Tampilan untuk pesan PENGGUNA (mungkin ada gambar)
                if "text" in content:
                    if content.get("image_bytes"):
                        st.image(content["image_bytes"], width=200, caption="Gambar yang Anda unggah")
                    st.markdown(content["text"])
                # Tampilan untuk respons ASISTEN (reasoning + grid produk)
                elif "reasoning" in content:
                    st.markdown(content["reasoning"])
                    if not content["recommendations_df"].empty:
                        display_product_grid(content["recommendations_df"], key_prefix="rag_multi")
                    else:
                        st.write("Tidak ada produk spesifik yang cocok untuk direkomendasikan.")

    # 2. Area Input (uploader dan chat input)
    uploaded_image = st.file_uploader("Unggah gambar untuk mencari produk serupa", type=["png", "jpg", "jpeg"])
    
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
                try:
                    # Logika Multimodal: Gunakan embedding gambar jika ada, jika tidak, gunakan teks
                    if image_bytes:
                        st.info("🔎 Mencari berdasarkan kemiripan gambar...")
                        from PIL import Image
                        import io
                        input_image = Image.open(io.BytesIO(image_bytes))
                        image_inputs = siglip_processor(images=[input_image], return_tensors="pt")
                        with torch.no_grad():
                            query_vector = siglip_model.get_image_features(**image_inputs).squeeze(0).tolist()
                        search_query = "SELECT * FROM products ORDER BY image_embedding <=> %s::vector LIMIT 10;"
                    else:
                        st.info("🔎 Mencari berdasarkan deskripsi teks...")
                        text_inputs = siglip_processor(text=[prompt_text], return_tensors="pt", padding="max_length")
                        with torch.no_grad():
                            query_vector = siglip_model.get_text_features(**text_inputs).squeeze(0).tolist()
                        search_query = "SELECT * FROM products ORDER BY text_embedding <=> %s::vector LIMIT 10;"

                    vector_string = str(query_vector)
                    context_df = pd.read_sql(search_query, conn, params=(vector_string,))
                    context_for_llm = context_df[['title', 'brand', 'price', 'product_details_clean']].to_string()

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat mengambil data dari database: {e}")
                    context_for_llm = "Tidak ada data yang ditemukan."

                # Buat Prompt JSON yang "sadar konteks"
                search_type_info = "berdasarkan gambar yang diunggah pengguna" if image_bytes else "berdasarkan deskripsi teks pengguna"
                
                final_prompt = f"""
                PERINTAH: Anda adalah API yang mengembalikan format JSON. Jangan menulis teks atau penjelasan lain di luar blok JSON.
                Tugas Anda adalah memberikan alasan dan merekomendasikan produk dari daftar KONTEKS. Pencarian produk ini dilakukan {search_type_info}.

                ATURAN FORMAT JSON:
                - Jawab HANYA dalam format JSON yang valid.
                - Objek JSON harus memiliki dua kunci: "reasoning" dan "recommendations".
                - "reasoning": (string) Penjelasan teks mengapa Anda merekomendasikan produk tersebut. Mulai kalimat Anda dengan menyatakan bahwa pencarian dilakukan berdasarkan gambar atau teks.
                - "recommendations": (list) Daftar produk yang direkomendasikan. Setiap item adalah objek JSON yang HANYA berisi kunci "title".

                ---
                KONTEKS PRODUK:
                {context_for_llm}
                ---
                PERTANYAAN PENGGUNA:
                {prompt_text}
                ---
                JSON OUTPUT:
                """
                
                # Panggil Model dan Proses Output
                try:
                    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
                    rag_model = GenerativeModel(selected_model)
                    rag_config = {"temperature": 0.0, "max_output_tokens": 8192}
                    
                    response_text = rag_model.generate_content(final_prompt, generation_config=rag_config).text
                    
                    import json, re
                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if not match: raise ValueError("Tidak ditemukan format JSON yang valid dalam respons AI.")
                    
                    parsed_response = json.loads(match.group(0))
                    reasoning = parsed_response.get("reasoning", "Berikut rekomendasi saya.")
                    recommended_titles = [rec.get("title") for rec in parsed_response.get("recommendations", [])]
                    
                    recommendations_df = context_df[context_df['title'].isin(recommended_titles)].copy()
                    
                    assistant_response_content = {
                        "reasoning": reasoning,
                        "recommendations_df": recommendations_df
                    }
                    # Tandai pesan user sebagai sudah diproses
                    st.session_state.rag_messages[-1]["processed"] = True
                    st.session_state.rag_messages.append({"role": "assistant", "content": assistant_response_content})
                    st.rerun()

                except Exception as e:
                    st.error(f"Gagal memproses respons dari AI: {e}")
                    st.code(response_text, language="text")