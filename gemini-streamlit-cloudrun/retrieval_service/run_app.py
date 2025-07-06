import os
from fastapi import FastAPI, Query
from typing import List, Optional
from pydantic import BaseModel
import psycopg2
from psycopg2.extras import RealDictCursor
from models.product_models import Product, ProductSearchResponse

# Load konfigurasi DB dari environment variable atau config.yml
DB_HOST = os.getenv("DB_HOST", "35.185.177.24")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "taragbagas")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "bagaswahyu")

app = FastAPI(title="Retrieval Service (RAG) - E-commerce")

def get_db_connection():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD
    )

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/products", response_model=ProductSearchResponse)
def search_products(
    query: Optional[str] = Query(None, description="Query teks (akan digunakan untuk semantic search embedding)"),
    limit: int = 10,
    offset: int = 0
):
    """
    Endpoint utama untuk search produk.
    - Jika query dikirim: lakukan semantic search berbasis vector (qwen3_embedding).
    - Jika tidak: tampilkan produk random/terbaru.
    """
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)

    # Semantic search dengan vektor (jika ada query, asumsikan embedding sudah di-generate dari frontend)
    # Untuk prototipe: query hanya by judul (bisa diganti search embedding jika embedding query diberikan dari frontend)
    if query:
        # Untuk demo: full-text search sederhana, bisa diganti dengan semantic vector search
        cur.execute(
            "SELECT * FROM products WHERE title ILIKE %s OR features_clean ILIKE %s LIMIT %s OFFSET %s",
            (f"%{query}%", f"%{query}%", limit, offset)
        )
    else:
        cur.execute(
            "SELECT * FROM products LIMIT %s OFFSET %s",
            (limit, offset)
        )

    rows = cur.fetchall()
    conn.close()
    products = [Product(**row) for row in rows]
    return ProductSearchResponse(products=products, total=len(products))

@app.get("/product/{product_id}", response_model=Product)
def get_product(product_id: int):
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    cur.execute("SELECT * FROM products WHERE product_id = %s", (product_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return {"error": "Product not found"}
    return Product(**row)

# Jika ingin semantic search berbasis vector:
# Bisa tambahkan endpoint POST yang menerima embedding query dan mengembalikan produk paling mirip
# (butuh pgvector di DB, dan input: embedding float[] dari frontend/chatbot)
