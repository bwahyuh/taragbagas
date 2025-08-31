# pylint: disable=broad-exception-caught,broad-exception-raised,invalid-name
"""
This module demonstrates the usage of the Gemini API in Vertex AI within a Streamlit application.
"""
import os
import io
import json
import re
import textwrap
import time

import google.auth
import httpx
import pandas as pd
import psycopg2
import streamlit as st
import torch
import vertexai
from google.genai.types import ThinkingConfig
from PIL import Image
from transformers import AutoModel, AutoProcessor
from vertexai.generative_models import GenerativeModel, Part

# --- Global Application Configuration ---
PROJECT_ID = "taragbagas-468109"
LOCATION = "us-central1"

# --- Vertex AI Initialization ---
try:
    credentials, _ = google.auth.default(
        scopes=["https://www.googleapis.com/auth/cloud-platform"]
    )
    vertexai.init(project=PROJECT_ID, location=LOCATION, credentials=credentials)
except Exception as e:
    st.error(f"Failed to initialize Vertex AI: {e}")
    st.stop()

# --- Database Connection ---
@st.cache_resource
def get_db_connection():
    """Gets a database connection and caches it for the session."""
    try:
        conn = psycopg2.connect(
            host="34.55.47.94",
            port="5432",
            database="postgres",
            user="postgres",
            password="Tamvan90."
        )
        return conn
    except psycopg2.OperationalError as e:
        st.error(
            f"🚨 Database Connection Error: Could not connect to PostgreSQL. "
            f"Please ensure the database is active and credentials are correct. Details: {e}"
        )
        st.stop()

# --- Google Cloud Helper Functions ---
def _project_id() -> str:
    """Use the Google Auth helper to get the Google Cloud Project."""
    try:
        _, project = google.auth.default()
    except google.auth.exceptions.DefaultCredentialsError as e:
        raise Exception("Could not automatically determine credentials") from e
    if not project:
        raise Exception("Could not determine project from credentials.")
    return project

def _region() -> str:
    """Use the local metadata service to get the region."""
    try:
        resp = httpx.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/region",
            headers={"Metadata-Flavor": "Google"},
        )
        return resp.text.split("/")[-1]
    except Exception:
        return "us-central1"

# --- Model Configuration ---
MODELS = {"gemini-2.5-pro": "Gemini 2.5 Pro"}
THINKING_BUDGET_MODELS = {"gemini-2.5-pro", "gemini-2.5-flash"}

# --- UI Helper Functions ---
@st.dialog("Product Details", width="large")
def show_product_details(shoe_data):
    """Displays full product details and a mock order button."""
    st.markdown(f"### {shoe_data['title']}")
    st.image(shoe_data['image_path'], use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Brand:**")
        st.markdown(f"**_{shoe_data['brand']}_**")
    with col2:
        st.markdown(f"**Price:**")
        st.markdown(f"**_{shoe_data['price']}_**")

    st.markdown("---")
    with st.expander("View Product Details"):
        st.write(shoe_data['product_details_clean'])
    with st.expander("Features"):
        st.write(shoe_data['features_clean'])
    st.markdown("---")

    order_col, close_col = st.columns([1, 1])
    with order_col:
        if st.button("🛒 Order Now", type="primary", use_container_width=True, key=f"order_{shoe_data['title']}"):
            st.toast(f"🎉 Your order for '{shoe_data['title']}' has been received!", icon="✅")
            time.sleep(2)
            st.rerun()
    with close_col:
        if st.button("Close", use_container_width=True, key=f"close_{shoe_data['title']}"):
            st.rerun()

def display_product_grid(df, key_prefix=""):
    """Takes a DataFrame and displays it in a 5-column grid format."""
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
                    if st.button("View Details", key=f"{key_prefix}_btn_{shoe['title']}_{idx}"):
                        show_product_details(shoe.to_dict())
                    st.markdown("---")

@st.cache_data
def get_random_products(_conn):
    """Fetches 20 random products to display and caches the result."""
    try:
        random_query = "SELECT * FROM products WHERE image_path IS NOT NULL ORDER BY RANDOM() LIMIT 20"
        return pd.read_sql(random_query, _conn)
    except Exception as e:
        st.warning(f"Could not load random products. Details: {e}")
        return pd.DataFrame()

# --- Model Loading ---
@st.cache_resource
def load_siglip_model():
    """Loads the SigLIP model and processor and caches them."""
    model_name = "google/siglip-so400m-patch14-384"
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32)
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

# --- Main Application UI ---
st.header("👟 Shoe Recommendation System", divider="rainbow")

# Load connection and models on startup
conn = get_db_connection()
siglip_model, siglip_processor = load_siglip_model()

tab_list = [
    "🛒 Keyword Search",
    "🗣️ Semantic Search (Text)",
    "📸 Image Search",
    "📸 + ✍️ Multimodal Search",
    "🤖 Smart Shopping Assistant"
]
tab1, tab2, tab3, tab4, tab5 = st.tabs(tab_list)

# --- TAB 1: KEYWORD SEARCH ---
with tab1:
    st.subheader("Search by Product Name")
    st.info("💡 Enter a partial shoe name (e.g., 'running shoes') to find recommendations.")

    selected_shoe_name = st.text_input("Recommend shoes similar to:", key="keyword_search")
    recommend_button = st.button("Recommend Me!", key="keyword_recommend_button")

    if 'keyword_recommendations' not in st.session_state:
        st.session_state.keyword_recommendations = None
    if 'keyword_matched_title' not in st.session_state:
        st.session_state.keyword_matched_title = None

    if recommend_button and selected_shoe_name:
        with st.spinner(f"Searching for products matching '{selected_shoe_name}'..."):
            try:
                cur = conn.cursor()
                like_query = "SELECT * FROM products WHERE LOWER(title) LIKE %s LIMIT 20"
                search_term = f"%{selected_shoe_name.lower()}%"
                cur.execute(like_query, (search_term,))
                rows = cur.fetchall()
                columns = [desc[0] for desc in cur.description]
                df_shoes = pd.DataFrame(rows, columns=columns)

                if df_shoes.empty:
                    st.warning("No shoes matching that name were found in the database.")
                    st.session_state.keyword_recommendations = pd.DataFrame()
                    st.session_state.keyword_matched_title = None
                else:
                    st.session_state.keyword_matched_title = selected_shoe_name
                    st.info(f"Showing search results for: '{selected_shoe_name}'")
                    st.session_state.keyword_recommendations = df_shoes
            except Exception as e:
                st.error(f"An error occurred while fetching data: {e}")
                st.session_state.keyword_recommendations = None
    elif recommend_button and not selected_shoe_name:
        st.session_state.keyword_recommendations = None

    if st.session_state.get('keyword_recommendations') is not None:
        if not st.session_state.keyword_recommendations.empty:
            st.success(f"Showing top recommendations for '{st.session_state.keyword_matched_title}':")
            display_product_grid(st.session_state.keyword_recommendations, key_prefix="keyword")
        else:
            st.warning("No products to display.")
    else:
        st.subheader("Explore Our Products")
        random_products = get_random_products(conn)
        display_product_grid(random_products, key_prefix="keyword_random")

# --- TAB 2: SEMANTIC SEARCH ---
with tab2:
    st.subheader("Search by Description (Text)")
    st.info("💡 Describe the product you're looking for, e.g., 'comfortable sandals for summer travel'.")
    
    semantic_search_term = st.text_area("Describe the product you are looking for:", key="semantic_search", height=100)
    semantic_recommend_button = st.button("Recommend Me!", key="semantic_recommend_button")

    if 'semantic_recommendations' not in st.session_state:
        st.session_state.semantic_recommendations = None

    if semantic_recommend_button and semantic_search_term:
        with st.spinner("Analyzing your description and searching for products..."):
            try:
                # **FIX**: Truncate long text inputs
                max_text_length = 300
                truncated_text = semantic_search_term[:max_text_length]
                
                inputs = siglip_processor(text=[truncated_text], return_tensors="pt", padding="max_length")
                with torch.no_grad():
                    input_vector = siglip_model.get_text_features(**inputs).squeeze(0).tolist()
                
                if input_vector:
                    vector_string = str(input_vector)
                    rec_query = "SELECT * FROM products WHERE text_embedding IS NOT NULL ORDER BY text_embedding <=> %s::vector ASC LIMIT 20"
                    df_shoes = pd.read_sql(rec_query, conn, params=(vector_string,))
                    st.session_state.semantic_recommendations = df_shoes
                else:
                    st.warning("Could not generate an embedding from your description.")
                    st.session_state.semantic_recommendations = pd.DataFrame()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.semantic_recommendations = pd.DataFrame()
    elif semantic_recommend_button and not semantic_search_term:
        st.session_state.semantic_recommendations = None

    if st.session_state.get('semantic_recommendations') is not None:
        st.success("Showing recommendations based on your description:")
        display_product_grid(st.session_state.semantic_recommendations, key_prefix="semantic")
    else:
        st.subheader("Explore Our Products")
        random_products = get_random_products(conn)
        display_product_grid(random_products, key_prefix="semantic_random")

# --- TAB 3: IMAGE SEARCH ---
with tab3:
    st.subheader("Search by Image")
    st.info("💡 Upload a product image to find visually similar items.")
    
    uploaded_image = st.file_uploader("Upload a product image...", type=["jpg", "jpeg", "png"], key="image_upload")
    multimodal_recommend_button = st.button("Recommend Me!", key="multimodal_recommend_button")

    if 'multimodal_recommendations' not in st.session_state:
        st.session_state.multimodal_recommendations = None

    if multimodal_recommend_button and uploaded_image is not None:
        with st.spinner("Analyzing the image and searching for similar products..."):
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
                    st.warning("Could not generate an embedding from your image.")
                    st.session_state.multimodal_recommendations = pd.DataFrame()
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.multimodal_recommendations = pd.DataFrame()
    elif multimodal_recommend_button and uploaded_image is None:
        st.warning("Please upload an image first.")
        st.session_state.multimodal_recommendations = None

    if st.session_state.get('multimodal_recommendations') is not None:
        st.success("Showing recommendations based on your image:")
        display_product_grid(st.session_state.multimodal_recommendations, key_prefix="multimodal")
    else:
        st.subheader("Explore Our Products")
        random_products = get_random_products(conn)
        display_product_grid(random_products, key_prefix="multimodal_random")

# --- TAB 4: MULTIMODAL SEARCH (Text + Image) ---
with tab4:
    st.subheader("Multimodal Search (Text + Image)")
    st.info("💡 Enter a product description and/or upload an image to find the best match.")

    multi_image = st.file_uploader(
        "Upload a product image:",
        type=["jpg", "jpeg", "png"],
        key="multi_image_upload_vertical"
    )
    if multi_image:
        st.image(multi_image, caption="Image preview", width=220)
    
    multi_text = st.text_area(
        "Describe the product:",
        key="multi_text_input_vertical",
        height=100
    )

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
        with st.spinner("Processing multimodal search..."):
            try:
                text_vector, image_vector = None, None
                rec_query, query_vector = None, None

                if multi_text:
                    # **FIX**: Truncate long text inputs
                    max_text_length = 300
                    truncated_text = multi_text[:max_text_length]
                    
                    text_inputs = siglip_processor(
                        text=[truncated_text], return_tensors="pt", padding="max_length"
                    )
                    with torch.no_grad():
                        text_vector = siglip_model.get_text_features(**text_inputs).squeeze(0)
                if multi_image is not None:
                    img = Image.open(multi_image).convert("RGB")
                    img_inputs = siglip_processor(images=[img], return_tensors="pt")
                    with torch.no_grad():
                        image_vector = siglip_model.get_image_features(**img_inputs).squeeze(0)
                
                if text_vector is not None and image_vector is not None:
                    query_vector = (text_vector + image_vector) / 2
                    rec_query = "SELECT * FROM products WHERE image_embedding IS NOT NULL ORDER BY image_embedding <=> %s::vector ASC LIMIT 20"
                elif text_vector is not None:
                    query_vector = text_vector
                    rec_query = "SELECT * FROM products WHERE text_embedding IS NOT NULL ORDER BY text_embedding <=> %s::vector ASC LIMIT 20"
                elif image_vector is not None:
                    query_vector = image_vector
                    rec_query = "SELECT * FROM products WHERE image_embedding IS NOT NULL ORDER BY image_embedding <=> %s::vector ASC LIMIT 20"
                
                if query_vector is not None:
                    vector_string = str(query_vector.tolist())
                    df_recs = pd.read_sql(rec_query, conn, params=(vector_string,))
                    st.session_state.multimodal_mix_recommendations = df_recs
                else:
                    st.warning("At least one input (text/image) is required!")
                    st.session_state.multimodal_mix_recommendations = pd.DataFrame()

            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.session_state.multimodal_mix_recommendations = pd.DataFrame()
    elif multi_search_button and (not multi_text and multi_image is None):
        st.warning("Please enter text, upload an image, or both!")
        st.session_state.multimodal_mix_recommendations = None

    if st.session_state.get('multimodal_mix_recommendations') is not None:
        st.success("Showing results from your multimodal search:")
        display_product_grid(st.session_state.multimodal_mix_recommendations, key_prefix="multimodal_mix")
    else:
        st.subheader("Explore Our Products")
        random_products = get_random_products(conn)
        display_product_grid(random_products, key_prefix="multimodal_mix_random")

# --- TAB 5: RAG-BASED CHAT APPLICATION ---
with tab5:
    st.subheader("🤖 Smart Shopping Assistant")
    selected_model = st.selectbox(
        "Select Model:",
        MODELS.keys(),
        format_func=lambda name: MODELS.get(name, "Gemini"),
        key="rag_selected_model"
    )

    st.info(
        f"You can ask questions or upload a product image to find similar items! "
        f"The currently active model is **{selected_model}**."
    )
    
    if "rag_messages" not in st.session_state:
        st.session_state.rag_messages = [{
            "role": "assistant",
            "content": "Hello! How can I help you with our shoe products today? You can ask a question or upload an image."
        }]
    
    for message_idx, message in enumerate(st.session_state.rag_messages):
        content = message["content"]
        with st.chat_message(message["role"]):
            if isinstance(content, str):
                st.markdown(content)
            elif isinstance(content, dict):
                if "text" in content:
                    if content.get("image_bytes"):
                        st.image(content["image_bytes"], width=200, caption="Your uploaded image")
                    st.markdown(content["text"])
                elif "reasoning" in content:
                    st.markdown(content["reasoning"])
                    if "recommendations_df" in content and not content["recommendations_df"].empty:
                        unique_key_prefix = f"rag_multi_{message_idx}"
                        display_product_grid(content["recommendations_df"], key_prefix=unique_key_prefix)
                    else:
                        st.write("No specific products found to recommend.")

    uploaded_image = st.file_uploader("Upload an image to find similar products", type=["png", "jpg", "jpeg"], key="rag_file_uploader")
    
    if prompt := st.chat_input("Write a question or describe the image..."):
        user_message_content = {"text": prompt}
        if uploaded_image is not None:
            user_message_content["image_bytes"] = uploaded_image.getvalue()
            user_message_content["mime_type"] = uploaded_image.type
        
        st.session_state.rag_messages.append({"role": "user", "content": user_message_content})
        st.rerun()

    last_message = st.session_state.rag_messages[-1]
    if last_message["role"] == "user" and not last_message.get("processed", False):
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                user_content = last_message["content"]
                prompt_text = user_content["text"]
                image_bytes = user_content.get("image_bytes")
                image_mime_type = user_content.get("mime_type")
                
                context_df = pd.DataFrame()
                context_for_llm = "[]"
                normalized_prompt_text = prompt_text.lower().replace("’", "'")
                image_content_description = "N/A"

                if image_bytes and image_mime_type:
                    try:
                        image_analysis_model = GenerativeModel(selected_model)
                        image_part = Part.from_data(data=image_bytes, mime_type=image_mime_type)
                        response = image_analysis_model.generate_content(
                            ["Describe the primary object in this image in one or two words.", image_part]
                        )
                        image_content_description = response.text.strip().lower()
                    except Exception as e:
                        st.warning(f"Could not analyze image: {e}")
                        image_content_description = "analysis failed"

                try:
                    text_vector, image_vector = None, None
                    if prompt_text:
                        # **FIX**: Truncate long text inputs
                        max_text_length = 300
                        truncated_text = prompt_text[:max_text_length]
                        
                        text_inputs = siglip_processor(text=[truncated_text], return_tensors="pt", padding="max_length")
                        with torch.no_grad():
                            text_vector = siglip_model.get_text_features(**text_inputs).squeeze(0)
                    if image_bytes:
                        input_image = Image.open(io.BytesIO(image_bytes))
                        image_inputs = siglip_processor(images=[input_image], return_tensors="pt")
                        with torch.no_grad():
                            image_vector = siglip_model.get_image_features(**image_inputs).squeeze(0)
                    
                    query_vector, search_query = None, None
                    if text_vector is not None and image_vector is not None:
                        query_vector = (text_vector + image_vector) / 2
                        # --- PERUBAHAN 1: Mengambil 10 produk kandidat ---
                        search_query = "SELECT * FROM products ORDER BY image_embedding <=> %s::vector LIMIT 10;"
                    elif image_vector is not None:
                        query_vector = image_vector
                        # --- PERUBAHAN 1: Mengambil 10 produk kandidat ---
                        search_query = "SELECT * FROM products ORDER BY image_embedding <=> %s::vector LIMIT 10;"
                    elif text_vector is not None:
                        query_vector = text_vector
                        # --- PERUBAHAN 1: Mengambil 10 produk kandidat ---
                        search_query = "SELECT * FROM products ORDER BY text_embedding <=> %s::vector LIMIT 10;"

                    if query_vector is not None and search_query:
                        vector_string = str(query_vector.tolist())
                        context_df = pd.read_sql(search_query, conn, params=(vector_string,))
                        if not context_df.empty:
                            temp_df = context_df.copy()
                            temp_df['title'] = temp_df['title'].str.lower().str.replace("’", "'", regex=False)
                            temp_df['brand'] = temp_df['brand'].str.lower().str.replace("’", "'", regex=False)
                            if 'features_clean' in temp_df.columns:
                                temp_df['features_clean'] = temp_df['features_clean'].astype(str).str.lower().str.replace("’", "'", regex=False)
                            context_for_llm = temp_df[['title', 'brand', 'price', 'features_clean']].to_json(orient='records', indent=2)

                except Exception as e:
                    st.error(f"An error occurred while fetching data from the database: {e}")
                
                # --- PERUBAHAN 2: Mengubah instruksi pada prompt ---
                final_prompt = f"""
                You are 'SoleMate', an expert and friendly AI shopping assistant for a high-end shoe store.
                Your primary goal is to help users find the perfect shoes. You must be conversational and handle various situations gracefully.
                Analyze the user's query and the retrieved PRODUCT CONTEXT, then follow these steps to generate a response in JSON format.

                **Step 1: Determine the User's Intent.**
                Based on the `USER'S INPUT`, classify the intent into one of three categories:
                1. `greeting_or_smalltalk`: The user is making a simple conversational gesture (e.g., "hello", "thanks", "how are you?").
                2. `product_query`: The user is asking about shoes, looking for recommendations, or describing a type of shoe.
                3. `off_topic_query`: The user's query is about something completely unrelated to footwear.

                **Step 2: Formulate a Response Based on the Intent.**
                Construct your JSON output according to the following rules:

                ---
                **RULE A: If intent is `greeting_or_smalltalk`:**
                - `response_type`: "greeting"
                - `message`: A warm, friendly, and brief conversational reply.

                ---
                **RULE B: If intent is `off_topic_query`:**
                - `response_type`: "off_topic"
                - `message`: Politely state that you are a shoe specialist and cannot help with non-footwear items.

                ---
                **RULE C: If intent is `product_query`:**
                - `response_type`: "recommendation"
                - Analyze the `PRODUCT CONTEXT`.
                - If context is empty or irrelevant, provide a `reasoning` message saying you couldn't find a match and an empty `recommendations` list.
                - If context is relevant:
                    - `reasoning`: A helpful, detailed explanation in Markdown. Start with a friendly opening paragraph. Then, for **EACH** product provided in the context, create a numbered list item with the following nested format:
                        1. **[Full Product Title]**
                           - **Query Match**: Briefly explain why this product is a good match for the user's request.
                           - **Key Features**: Based on the `features_clean` column, write a detailed 2-4 sentence description summarizing the most prominent features.
                    - `recommendations`: A list of JSON objects, each with the exact `title` (lowercase) of **ALL** products from the `PRODUCT CONTEXT`.

                ---
                **PRODUCT CONTEXT (from database vector search):**
                {context_for_llm}

                **USER'S INPUT:**
                - **Image Content Description**: {image_content_description}
                - **User Text**: {normalized_prompt_text}

                **JSON OUTPUT:**
                """
                
                response_text = "" # Initialize to prevent NameError
                try:
                    rag_model = GenerativeModel(selected_model)
                    rag_config = {
                        "temperature": 0.1, 
                        "max_output_tokens": 8192,
                        "top_p": 0.95
                    }
                    
                    response = rag_model.generate_content(final_prompt, generation_config=rag_config)
                    response_text = response.text
                    
                    match = re.search(r'\{.*\}', response_text, re.DOTALL)
                    if not match: raise ValueError("No valid JSON format found in the AI's response.")
                    
                    parsed_response = json.loads(match.group(0))
                    response_type = parsed_response.get("response_type")

                    # --- Parsing logic remains the same ---
                    if response_type == "recommendation":
                        reasoning = parsed_response.get("reasoning", "Here are my recommendations.")
                        recommendations_list = parsed_response.get("recommendations", [])
                        
                        recommended_titles_lower = []
                        if recommendations_list and isinstance(recommendations_list[0], (dict, str)):
                            if isinstance(recommendations_list[0], dict):
                                recommended_titles_lower = [rec.get("title") for rec in recommendations_list if rec.get("title")]
                            else: # It's a list of strings
                                recommended_titles_lower = recommendations_list
                        
                        recommendations_df = pd.DataFrame()
                        if not context_df.empty and recommended_titles_lower:
                            recommendations_df = context_df[context_df['title'].str.lower().str.replace("’", "'", regex=False).isin(recommended_titles_lower)].copy()
                        
                        assistant_response_content = {
                            "reasoning": reasoning,
                            "recommendations_df": recommendations_df
                        }
                    else: # Handles "greeting" and "off_topic"
                        message = parsed_response.get("message", "I'm not sure how to respond to that.")
                        assistant_response_content = message 

                    st.session_state.rag_messages[-1]["processed"] = True
                    st.session_state.rag_messages.append({"role": "assistant", "content": assistant_response_content})
                    st.rerun()

                except Exception as e:
                    st.error(f"Failed to process the response from the AI: {e}")
                    with st.expander("View Debug Details"):
                        st.code(final_prompt, language="markdown")
                        st.code(response_text, language="text")
