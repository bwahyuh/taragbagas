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
    qwen3_embedding vector(1024),     -- untuk Qwen3-embedding-0.6B
    image_embedding vector(1152),     -- untuk siglip
    image_path TEXT
);
