import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd


class NeuralCF(nn.Module):
    """
    A simple Neural Collaborative Filtering model using matrix factorization.
    """
    def __init__(self, num_users, num_items, emb_size=32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, emb_size)
        self.item_emb = nn.Embedding(num_items, emb_size)
        self.fc = nn.Linear(emb_size * 2, 1)

    def forward(self, users, items):
        u = self.user_emb(users)
        i = self.item_emb(items)
        x = torch.cat([u, i], dim=1)
        # Aktivasyon ekledik → skorları 0-1 aralığına çek
        return torch.sigmoid(self.fc(x)).squeeze()


def train_ncf(df, epochs=5, lr=1e-3, batch_size=1024, emb_size=32):
    """
    Trains the Neural Collaborative Filtering model.

    Args:
        df (pd.DataFrame): DataFrame with 'user_id', 'product_id', and 'rating'.
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        batch_size (int): Batch size for training.
        emb_size (int): Size of the user and item embeddings.

    Returns:
        tuple: Trained model and mappings for user/item IDs.
    """
    df2 = df.copy()
    
    # Convert user/product IDs to categorical indices
    df2['user_idx'] = df2['user_id'].astype('category').cat.codes
    df2['item_idx'] = df2['product_id'].astype('category').cat.codes
    # Normalize ratings to 0–1 range
    df2['rating'] = (df2['rating'] - df2['rating'].min()) / (df2['rating'].max() - df2['rating'].min())


    num_users = df2['user_idx'].nunique()
    num_items = df2['item_idx'].nunique()

    model = NeuralCF(num_users, num_items, emb_size=emb_size)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    # Convert to PyTorch Tensors
    users = torch.LongTensor(df2['user_idx'].values)
    items = torch.LongTensor(df2['item_idx'].values)
    ratings = torch.FloatTensor(df2['rating'].values)

    dataset = torch.utils.data.TensorDataset(users, items, ratings)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for u, i, r in loader:
            optimizer.zero_grad()
            pred = model(u, i)
            loss = loss_fn(pred, r)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * u.size(0)
        
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss / len(df2):.4f}")

    # Create mappings for future predictions
    user_map = dict(enumerate(df2['user_id'].astype('category').cat.categories))
    item_map = dict(enumerate(df2['product_id'].astype('category').cat.categories))
    inv_user_map = {v: k for k, v in user_map.items()}
    inv_item_map = {v: k for k, v in item_map.items()}

    return model, inv_user_map, inv_item_map


def ncf_predict_user(model, inv_user_map, inv_item_map, user_id, df_products):
    """
    Generates recommendations for a specific user using the NCF model.
    """
    model.eval()
    if user_id not in inv_user_map:
        print(f"User ID {user_id} not found in the model's user map.")
        return pd.DataFrame()

    # User index
    uidx = inv_user_map[user_id]

    # Item indekslerini al
    item_indices = list(inv_item_map.values())  # bunlar integer indeksler
    item_ids = list(inv_item_map.keys())        # bunlar gerçek product_id’ler

    # Prediction
    user_tensor = torch.LongTensor([uidx] * len(item_indices))
    item_tensor = torch.LongTensor(item_indices)

    with torch.no_grad():
        preds = model(user_tensor, item_tensor).detach().cpu().numpy()

    # Doğru eşleştirme: aynı sırayla product_id + skor
    result_df = pd.DataFrame({
        'product_id': item_ids,
        'ncf_score': preds
    }).sort_values('ncf_score', ascending=False)

    return result_df
