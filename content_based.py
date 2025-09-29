import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import BertTokenizer, BertModel
import torch

class ContentModel:
    def __init__(self):
        # BERT modelini ve tokenizer'ı önceden yükle
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.product_embeddings = None
        self.product_index = None

    def fit(self, df_products, text_col='text'):
        texts = df_products[text_col].fillna("").tolist()
        
        # Ürün metinlerini BERT'in anlayacağı şekilde token'lara ayır
        # 'cls' token'ını (ilk token) kullanarak cümle embedding'ini alırız
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        
        # Metinleri BERT ile embedding'lere dönüştür
        with torch.no_grad():
            outputs = self.model(**inputs)
            # İlk token'ın (CLS) embedding'ini alırız.
            # Bu, tüm cümlenin bir özetidir.
            self.product_embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        # Ürün kimliklerini indeksleme için bir sözlük oluştur
        self.product_index = pd.Series(df_products.index.values, index=df_products['product_id']).to_dict()

    def recommend(self, target_product_id, df_products, top_n=5):
        if target_product_id not in self.product_index:
            return pd.DataFrame()
        
        idx = self.product_index[target_product_id]
        
        # Hedef ürünün embedding'ini al
        target_embedding = self.product_embeddings[idx].reshape(1, -1)
        
        # Tüm ürünlerle kosinüs benzerliği hesapla
        sims = cosine_similarity(target_embedding, self.product_embeddings).flatten()
        scores = pd.Series(sims, index=df_products.index)
        scores = scores.drop(idx, errors='ignore')
        
        top_indices = scores.sort_values(ascending=False).head(top_n).index
        recommendations = df_products.loc[top_indices].copy()
        recommendations['cb_score'] = scores.loc[top_indices].values
        
        return recommendations.reset_index(drop=True)