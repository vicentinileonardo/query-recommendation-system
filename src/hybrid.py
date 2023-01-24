import os
import pandas as pd


DIR = os.path.dirname(__file__)

def import_data(matrix_type="partial"):
    if matrix_type == "partial":
        path = os.path.join(DIR, '../data/_utility_matrix.csv')
    elif matrix_type == "real_complete":
        path = os.path.join(DIR, '../data/_utility_matrix_complete.csv')
    else:
        print('Wrong matrix type!')
        return
    utility_matrix = pd.read_csv(path, index_col=0)
    return utility_matrix

def hybrid_recommender():

    real_complete_utility_matrix = import_data(matrix_type="real_complete")

    path_1 = os.path.join(DIR, '../data/item_item_cf/complete_utility_matrix.csv')
    path_2 = os.path.join(DIR, '../data/movies_item_item_cf/complete_utility_matrix.csv')
    item_item_CF_utility_matrix = pd.read_csv(path_1, index_col=0)
    movies_item_item_CF_utilty_matrix = pd.read_csv(path_2, index_col=0)

    THRESHOLD = 10

    query_results_length = 10 # how to calculate this?

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



