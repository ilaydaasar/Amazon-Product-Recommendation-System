import pandas as pd
import random

def generate_candidates(user_id, df_products, user_history, top_k=100):
    """
    Kullanıcının geçmişine ve popüler ürünlere dayalı aday listesi oluşturur.
    Popüler ürünlerden rastgele örnek alarak çeşitliliği artırır.
    """
    candidates = set()
    if user_id in user_history:
        candidates.update(user_history[user_id])
    
    # Popüler ürünleri al
    popular = df_products.sort_values('avg_rating', ascending=False).head(top_k)['product_id'].tolist()
    
    # Rastgele örnekle 50 tanesini al
    popular_sample = random.sample(popular, min(50, len(popular)))
    candidates.update(popular_sample)
    
    # Kullanıcının geçmiş ürünlerini çıkar
    if user_id in user_history:
        candidates = candidates - set(user_history[user_id])
        
    return list(candidates)


def score_and_rerank(candidates, df_products, content_model=None, target_product_id=None, ncf_df=None, top_n=10):
    """
    Aday ürünleri puanlar ve yeniden sıralar. NCF skorunu normalize etmeden kullanır.
    """
    df_c = df_products.set_index('product_id').loc[
        df_products.set_index('product_id').index.intersection(candidates)
    ].copy()

    # Puan sütunları
    df_c['cb_score'] = 0.0
    df_c['rating_score'] = df_c['avg_rating'] / (df_c['avg_rating'].max() if df_c['avg_rating'].max() > 0 else 1)
    df_c['ncf_score'] = 0.0

    # CB skorları
    if content_model and target_product_id:
        cb_recs = content_model.recommend(target_product_id, df_products, top_n=len(candidates))
        df_c = df_c.merge(cb_recs[['product_id', 'cb_score']], on='product_id', how='left', suffixes=('', '_new')).fillna(0)
        df_c['cb_score'] = df_c['cb_score_new']
        df_c.drop(columns=['cb_score_new'], inplace=True)

    # NCF skorları
    if ncf_df is not None and not ncf_df.empty:
        df_c = df_c.merge(ncf_df[['product_id', 'ncf_score']], on='product_id', how='left', suffixes=('', '_new')).fillna(0)
        df_c['ncf_score'] = df_c['ncf_score_new']
        df_c.drop(columns=['ncf_score_new'], inplace=True)
        # Normalize etme kısmı kaldırıldı

    # Ağırlıklı toplam score
    df_c['score'] = 0.4 * df_c['cb_score'] + 0.3 * df_c['ncf_score'] + 0.3 * df_c['rating_score']

    # Sırala ve döndür
    df_c = df_c.sort_values('score', ascending=False).reset_index()
    return df_c[['product_id', 'product_name', 'category', 'score', 'cb_score', 'rating_score', 'ncf_score']]
