from item_item_CF import get_top_k_queries, save_top_k_queries

def hybrid_recommender(item_item_CF_utility_matrix, movies_item_item_CF_utilty_matrix):

    item_item_CF_weight = 0.5
    movies_item_item_CF_weight = 0.5

    '''
    THRESHOLD = 10
    if query_results < THRESHOLD:
        item_item_CF_weight = 0.2
        movies_item_item_CF_weight = 0.8
    else:
        item_item_CF_weight = 0.5
        movies_item_item_CF_weight = 0.5
    '''

    hybrid_utility_matrix = item_item_CF_weight * item_item_CF_utility_matrix + movies_item_item_CF_weight * movies_item_item_CF_utilty_matrix

    return hybrid_utility_matrix





