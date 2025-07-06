# embedding_utils/embedding_tools.py

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
from PIL import Image
from transformers import AutoProcessor, AutoModel as ImageModel

# --- SETUP: Path model HuggingFace ---
QWEN3_MODEL_NAME = "Qwen/Qwen3-embedding-0.6B"
SIGLIP_MODEL_NAME = "google/siglip-so400m-patch14-384"

# --- TEKS EMBEDDING: Qwen3 ---
class Qwen3Embedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_NAME)
        self.model = AutoModel.from_pretrained(QWEN3_MODEL_NAME).to(self.device)

    def embed(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        return embedding.tolist()  # Output: list of float (len=1024)

# --- GAMBAR EMBEDDING: SigLIP ---
class SigLIPEmbedder:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_NAME)
        self.model = ImageModel.from_pretrained(SIGLIP_MODEL_NAME).to(self.device)

    def embed(self, image_path: str):
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            embedding = outputs.image_embeds.squeeze().cpu().numpy()
        return embedding.tolist()  # Output: list of float (len=1152)

# --- Example usage ---
if __name__ == "__main__":
    qwen = Qwen3Embedder()
    text_embed = qwen.embed("Contoh kalimat produk sepatu nyaman dan ringan")
    print("Qwen3 Embedding dimension:", len(text_embed), "\n", text_embed[:10], "...")

    siglip = SigLIPEmbedder()
    # Misal ada file lokal 'sample.jpg'
    # image_embed = siglip.embed("sample.jpg")
    # print("SigLIP Embedding dimension:", len(image_embed), "\n", image_embed[:10], "...")
