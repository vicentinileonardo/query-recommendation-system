from item_item_CF import get_top_k_queries, save_top_k_queries

def hybrid_recommender(CF_utility_matrix, CB_utilty_matrix, CF_weight=0.50, CB_weight=0.50):

    hybrid_utility_matrix = CF_weight * CF_utility_matrix + CB_weight * CB_utilty_matrix

    return hybrid_utility_matrix





