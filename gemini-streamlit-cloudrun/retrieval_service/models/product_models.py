from pydantic import BaseModel, Field
from typing import Optional, List

class Product(BaseModel):
    product_id: Optional[int] = Field(None, description="ID unik produk (auto increment dari database)")
    title: str
    price: str
    brand: str
    product_details_clean: Optional[str] = None
    features_clean: Optional[str] = None
    breadcrumbs_clean: Optional[str] = None
    qwen3_embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None
    image_path: Optional[str] = None

class ProductSearchResponse(BaseModel):
    products: List[Product]
    total: int
