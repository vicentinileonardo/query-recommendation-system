from item_item_CF import get_top_k_queries, save_top_k_queries

def hybrid_recommender(item_item_CF_utility_matrix, movies_item_item_CF_utilty_matrix):

    THRESHOLD = 10

    item_item_CF_base_weight = 0.5
    movies_item_item_CF_base_weight = 0.5

    query_results_length = 10


    if query_results_length < THRESHOLD:
        item_item_CF_weight = 0.2
        movies_item_item_CF_weight = 0.8
    else:
        item_item_CF_weight = 0.5
        movies_item_item_CF_weight = 0.5


    hybrid_utility_matrix = item_item_CF_weight * item_item_CF_utility_matrix + movies_item_item_CF_weight * movies_item_item_CF_utilty_matrix

    return hybrid_utility_matrix


'''
import numpy as np

def hybrid_recommender(item_item_CF_utility_matrix, movies_item_item_CF_utilty_matrix, real_utility_matrix, item_item_CF_weight=0.5, movies_item_item_CF_weight=0.5, learning_rate=0.1, num_iterations=100):
    for i in range(num_iterations):
        # Calculate the current prediction
        hybrid_utility_matrix = item_item_CF_weight * item_item_CF_utility_matrix + movies_item_item_CF_weight * movies_item_item_CF_utilty_matrix
        
        # Calculate the error
        error = np.sum((real_utility_matrix - hybrid_utility_matrix) ** 2)
        
        # Calculate the gradient
        item_item_CF_weight_gradient = -2 * np.sum((real_utility_matrix - hybrid_utility_matrix) * item_item_CF_utility_matrix)
        movies_item_item_CF_weight_gradient = -2 * np.sum((real_utility_matrix - hybrid_utility_matrix) * movies_item_item_CF_utilty_matrix)
        
        # Update the weights
        item_item_CF_weight -= learning_rate * item_item_CF_weight_gradient
        movies_item_item_CF_weight -= learning_rate * movies_item_item_CF_weight_gradient
        
        print(f'Iteration: {i}, Error: {error}, item_item_CF_weight: {item_item_CF_weight}, movies_item_item_CF_weight: {movies_item_item_CF_weight}')
    return item_item_CF_weight, movies_item_item_CF_weight


'''



