import torch

def recall_at_k(num_user,num_item, K, model, test_dict, batches=5):
    """
    Compute Recall@K for a recommendation model.

    Parameters:
    - num_user: int, total number of users
    - num_item: int, total number of items
    - K: int, top-K items to consider
    - model: PyTorch model, should output scores for all items given a user
    - test_dict: dict, mapping user_id -> list of ground truth item_ids
    - batches: int, number of user batches to process at a time

    Returns:
    - recall: float, average Recall@K over all users
    """

    user_indices = list(range(num_user))
    USER_BATCH = num_user // batches
    recalls = []
    movie_indicies = list(range(num_item))
    for b in range(batches):
        batch_users = user_indices[b * USER_BATCH : (b + 1) * USER_BATCH]
        batch_users_tensor = torch.tensor(batch_users)

        # Model prediction for all items for these users
        with torch.no_grad():
            user_emb, movie_emb, _ = model(batch_users_tensor,movie_indicies,[])  # shape: [batch_size, num_item]
            scores = user_emb @ movie_emb.T
        # Get top-K item indices
        topk_items = torch.topk(scores, K, dim=1).indices

        # Compute recall per user
        for i, u in enumerate(batch_users):
            true_items = set(test_dict.get(u, []))
            if not true_items:
                continue
            recommended_items = set(topk_items[i].tolist())
            recalls.append(len(true_items & recommended_items) / len(true_items))

    # Average recall over all users
    return sum(recalls) / len(recalls)
