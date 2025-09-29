from data_preprocessing import load_and_prepare
from content_based import ContentModel
from collaborative import CFModel
from neural_cf import train_ncf, ncf_predict_user
from hybrid import generate_candidates, score_and_rerank
import pandas as pd
import random

# 1) Veri yükle
train_df, test_df, df_products, user_item_matrix = load_and_prepare('data/amazon.csv')

# 2) Content-based model
content = ContentModel()
content.fit(df_products, text_col='text')

# 3) CF model
cf = CFModel(n_components=20)
cf.fit(user_item_matrix)

# 4) NCF model
ncf_df = None
if 'rating' in train_df.columns and not train_df['rating'].isnull().all():
    ncf_model, inv_user_map, inv_item_map = train_ncf(train_df, epochs=15,emb_size=64, lr=5e-4)
    ncf_users = list(inv_user_map.keys())
else:
    print("NCF atlandı")
    ncf_users = []

# 5) Kullanıcı geçmişi
user_history = train_df.groupby('user_id')['product_id'].apply(list).to_dict()

# 6) Test için rastgele kullanıcılar seç (sadece NCF’de olan kullanıcılar)
test_users = random.sample(ncf_users, min(5, len(ncf_users))) if ncf_users else random.sample(list(test_df['user_id'].unique()), min(5, len(test_df['user_id'].unique())))

# 7) Dosyayı bir kez açıp tüm önerileri yaz
with open("recommendations.txt", "w", encoding="utf-8") as f:
    for sample_user in test_users:
        # NCF skorlarını al
        ncf_user_df = None
        if sample_user in ncf_users:
            ncf_user_df = ncf_predict_user(ncf_model, inv_user_map, inv_item_map, sample_user, df_products)

        # Referans ürün
        user_ratings = train_df[train_df['user_id'] == sample_user].sort_values('rating', ascending=False)
        sample_product = user_ratings['product_id'].iloc[0] if not user_ratings.empty else df_products.sort_values('avg_rating', ascending=False)['product_id'].iloc[0]

        # Aday ürünler
        candidates = generate_candidates(sample_user, df_products, user_history, top_k=500)

        # Önerileri al
        recommendations = score_and_rerank(candidates, df_products, content_model=content,
                                           target_product_id=sample_product, ncf_df=ncf_user_df, top_n=10)

        # Ekrana yazdır
        print(f"\nRastgele seçilen kullanıcı: {sample_user}")
        print(f"Referans ürün: {sample_product}")
        print("Hibrit öneriler (score ile birlikte):")
        print(recommendations)

        # Dosyaya yaz
        f.write(f"\nRastgele seçilen kullanıcı: {sample_user}\n")
        f.write(f"Referans ürün: {sample_product}\n")
        f.write("Hibrit öneriler (score ile birlikte):\n")
        f.write(recommendations.to_string(index=False))
        f.write("\n" + "="*50 + "\n")

print("\nTest önerileri 'recommendations.txt' dosyasına kaydedildi.")
