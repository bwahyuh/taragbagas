-- Skema tabel siap RAG
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE products (
    product_id SERIAL PRIMARY KEY,
    title TEXT,
    price TEXT,
    brand TEXT,
    product_details_clean TEXT,
    features_clean TEXT,
    breadcrumbs_clean TEXT,
    text_embedding vector(1152),     -- untuk siglip txt
    image_embedding vector(1152),     -- untuk siglip img
    image_path TEXT,
    combined_text TEXT
);
