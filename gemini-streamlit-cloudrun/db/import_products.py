import pandas as pd
import psycopg2
import ast

conn = psycopg2.connect(
    host="35.185.177.24", database="taragbagas", user="postgres", password="bagaswahyu", port=5432
)
cur = conn.cursor()
df = pd.read_csv("../data/Data-Final-cleaned-public-url-FIXED.csv")

for _, row in df.iterrows():
    qwen_vec = [float(x) for x in row['qwen3_embedding'].split(',')]
    img_vec = ast.literal_eval(row['image_embedding'])
    cur.execute(
        "INSERT INTO products (title, price, brand, product_details_clean, features_clean, breadcrumbs_clean, qwen3_embedding, image_embedding, image_path) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)",
        (
            row['title'], row['price'], row['brand'], row['product_details_clean'], row['features_clean'],
            row['breadcrumbs_clean'], qwen_vec, img_vec, row['image_path']
        )
    )
conn.commit()
cur.close()
conn.close()
