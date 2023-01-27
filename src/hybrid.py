import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
import torch


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


def log_to_txt(path, text):
    with open(path, 'a') as f:
        f.write(text)


def calculate_mae(real_utility_matrix_complete, utility_matrix_complete):
    mae = mean_absolute_error(real_utility_matrix_complete, utility_matrix_complete)
    return mae

def calculate_rmse(real_utility_matrix_complete, utility_matrix_complete):
    rmse = math.sqrt(mean_squared_error(real_utility_matrix_complete, utility_matrix_complete))
    return rmse

# epsilon smoothing due to a runtime error in dividing by zero or by a very small number
def calculate_mape(real_utility_matrix_complete, utility_matrix_complete, epsilon=1e-8):
    actual, predicted = np.array(real_utility_matrix_complete), np.array(utility_matrix_complete)
    mask = actual != 0
    return (np.fabs(actual - predicted)/(actual + epsilon))[mask].mean() * 100

def calculate_mre(real_utility_matrix_complete, utility_matrix_complete, epsilon=1e-8):
    actual, predicted = np.array(real_utility_matrix_complete), np.array(utility_matrix_complete)
    mask = actual != 0
    return (np.fabs(actual - predicted)/(actual + epsilon))[mask].mean()

def hybrid_recommender(N_QUERIES = 100, N_USERS = 2500, THRESHOLD_1 = 10, THRESHOLD_2 = 200, WEIGHT_1 = 0.7, WEIGHT_2 = 0.5, WEIGHT_3 = 0.3):

    real_complete_utility_matrix = import_data(matrix_type="real_complete")

    path_1 = os.path.join(DIR, '../data/item_item_cf/complete_utility_matrix.csv')
    path_2 = os.path.join(DIR, '../data/movies_item_item_cf/complete_utility_matrix.csv')
    item_item_CF_utility_matrix = pd.read_csv(path_1, index_col=0)
    movies_item_item_CF_utilty_matrix = pd.read_csv(path_2, index_col=0)

    path_preprocessed_queries = os.path.join(DIR, '../data/movies_item_item_cf/preprocessed_queries.csv')
    preprocessed_queries_df = pd.read_csv(path_preprocessed_queries, index_col=0)


    queries=real_complete_utility_matrix.head(N_QUERIES).columns.values
    users=real_complete_utility_matrix.head(N_USERS).index.values


    results_in_query={}

    for query in queries:
        try:
            results_in_query[query]=int(preprocessed_queries_df.loc[query].value_counts())
        except:
            results_in_query[query]=0

    hybrid_utility_matrix = pd.DataFrame(index=users, columns=queries)

    for query in queries:
        if results_in_query[query] > THRESHOLD_2:
            item_item_CF_weight = WEIGHT_1
            movies_item_item_CF_weight = 1 - WEIGHT_1
        elif results_in_query[query] < THRESHOLD_2 and results_in_query[query] > THRESHOLD_1:
            item_item_CF_weight = WEIGHT_2
            movies_item_item_CF_weight = 1 - WEIGHT_2
        else:
            item_item_CF_weight = WEIGHT_3
            movies_item_item_CF_weight = 1 - WEIGHT_3
        for user in users:
            hybrid_utility_matrix.loc[user,query] = round(item_item_CF_weight * item_item_CF_utility_matrix.loc[user,query] + movies_item_item_CF_weight * movies_item_item_CF_utilty_matrix.loc[user,query])
            #print(hybrid_utility_matrix.loc[user,query])

    hybrid_path = os.path.join(DIR, '../data/hybrid/complete_utility_matrix.csv')
    hybrid_utility_matrix.to_csv(hybrid_path)



    #return hybrid_utility_matrix

def hybrid_recommender_gd(N_QUERIES = 100, N_USERS = 2500, THRESHOLD_1 = 10, THRESHOLD_2 = 200, WEIGHT_1 = 0.7, WEIGHT_2 = 0.5, WEIGHT_3 = 0.3):

    real_complete_utility_matrix = import_data(matrix_type="real_complete")

    path_1 = os.path.join(DIR, '../data/item_item_cf/complete_utility_matrix.csv')
    path_2 = os.path.join(DIR, '../data/movies_item_item_cf/complete_utility_matrix.csv')
    item_item_CF_utility_matrix = pd.read_csv(path_1, index_col=0)
    movies_item_item_CF_utilty_matrix = pd.read_csv(path_2, index_col=0)

    path_preprocessed_queries = os.path.join(DIR, '../data/movies_item_item_cf/preprocessed_queries.csv')
    preprocessed_queries_df = pd.read_csv(path_preprocessed_queries, index_col=0)


    queries=real_complete_utility_matrix.head(N_QUERIES).columns.values
    users=real_complete_utility_matrix.head(N_USERS).index.values

    results_in_query={}

    for query in queries:
        try:
            results_in_query[query]=int(preprocessed_queries_df.loc[query].value_counts())
        except:
            results_in_query[query]=0

    hybrid_utility_matrix = pd.DataFrame(index=users, columns=queries)

    # brute force to find the best weights and thresholds
    best_mae = 100
    best_rmse = 100
    #best_mape = 0

    bf_PATH = os.path.join(DIR, '../data/hybrid/bf.txt')
    bf_best_PATH = os.path.join(DIR, '../data/hybrid/bf_best.txt')

    counter = 0
    weight_2 = 0.5
    for threshold_1 in range(1, 500, 100):
        for threshold_2 in range(500, 2500, 250):
            for weight_1 in np.arange(0.1, 1, 0.15):
                #for weight_2 in np.arange(0.1, 1, 0.15):
                for weight_3 in np.arange(1, 0.1, -0.15):
                    for query in queries:
                        if results_in_query[query] >= threshold_2:
                            item_item_CF_weight = weight_1
                            movies_item_item_CF_weight = 1 - weight_1
                        elif results_in_query[query] < threshold_2 and results_in_query[query] > threshold_1:
                            item_item_CF_weight = weight_2
                            movies_item_item_CF_weight = 1 - weight_2
                        else:
                            item_item_CF_weight = weight_3
                            movies_item_item_CF_weight = 1 - weight_3
                        for user in users:
                            hybrid_utility_matrix.loc[user,query] = round(item_item_CF_weight * item_item_CF_utility_matrix.loc[user,query] + movies_item_item_CF_weight * movies_item_item_CF_utilty_matrix.loc[user,query])
                    mae = calculate_mae(real_complete_utility_matrix, hybrid_utility_matrix)
                    rmse = calculate_rmse(real_complete_utility_matrix, hybrid_utility_matrix)
                    #mape = calculate_mape(real_complete_utility_matrix, hybrid_utility_matrix)
                    counter += 1
                    if counter % 50 == 0:
                        print(counter, "iterations done")
                        print(threshold_1, threshold_2, weight_1, weight_2, weight_3, mae, rmse)
                        log_to_txt(bf_PATH, 'Counter: ' + str(counter) + ' Thresholds: ' + str(threshold_1) + ' ' + str(threshold_2) + ' Weights: ' + str(weight_1) + ' ' + str(weight_2) + ' ' + str(weight_3) + ' MAE: ' + str(mae) + ' RMSE: ' + str(rmse) + '\n')

                    if mae < best_mae and rmse < best_rmse:
                        best_mae = mae
                        best_rmse = rmse
                        #best_mape = mape
                        best_threshold_1 = threshold_1
                        best_threshold_2 = threshold_2
                        best_weight_1 = weight_1
                        best_weight_2 = weight_2
                        best_weight_3 = weight_3
                        print("MAE: ", best_mae, "RMSE: ", best_rmse, "Threshold 1: ", best_threshold_1, "Threshold 2: ", best_threshold_2, "Weight 1: ", best_weight_1, "Weight 2: ", best_weight_2, "Weight 3: ", best_weight_3)
                        log_to_txt(bf_best_PATH, 'Counter: ' + str(counter)  +  ' MAE: ' + str(best_mae) + ' RMSE: ' + str(best_rmse) + ' Threshold 1: ' + str(best_threshold_1) + ' Threshold 2: ' + str(best_threshold_2) + ' Weight 1: ' + str(best_weight_1) + ' Weight 2: ' + str(best_weight_2) + ' Weight 3: ' + str(best_weight_3) + '\n')


    print('End of the search, best results are:')
    print("MAE: ", best_mae, "RMSE: ", best_rmse, "Threshold 1: ", best_threshold_1, "Threshold 2: ", best_threshold_2, "Weight 1: ", best_weight_1, "Weight 2: ", best_weight_2, "Weight 3: ", best_weight_3)

    for query in queries:
        if results_in_query[query] > best_threshold_2:
            item_item_CF_weight = best_weight_1
            movies_item_item_CF_weight = 1 - best_weight_1
        elif results_in_query[query] < best_threshold_2 and results_in_query[query] > best_threshold_1:
            item_item_CF_weight = best_weight_2
            movies_item_item_CF_weight = 1 - best_weight_2
        else:
            item_item_CF_weight = best_weight_3
            movies_item_item_CF_weight = 1 - best_weight_3
        for user in users:
            hybrid_utility_matrix.loc[user,query] = round(item_item_CF_weight * item_item_CF_utility_matrix.loc[user,query] + movies_item_item_CF_weight * movies_item_item_CF_utilty_matrix.loc[user,query])

    hybrid_path = os.path.join(DIR, '../data/hybrid/complete_utility_matrix.csv')
    hybrid_utility_matrix.to_csv(hybrid_path)



    return best_threshold_1, best_threshold_2, best_weight_1, best_weight_2, best_weight_3



'''
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


if __name__ == '__main__':

    N_QUERIES = 100
    N_USERS = 2500
    N_ITEMS = 7669

    THRESHOLD_1 = 200.0
    THRESHOLD_2 = 1000.0
    WEIGHT_1 = 0.7
    WEIGHT_2 = 0.5
    WEIGHT_3 = 0.3

    #hybrid_recommender(N_QUERIES = 100, N_USERS = 2500, THRESHOLD_1 = THRESHOLD_1, THRESHOLD_2 = THRESHOLD_2, WEIGHT_1 = WEIGHT_1, WEIGHT_2 = WEIGHT_2, WEIGHT_3 = WEIGHT_3)
    best_threshold_1, best_threshold_2, best_weight_1, best_weight_2, best_weight_3 = hybrid_recommender_gd(N_QUERIES = 100, N_USERS = 2500, THRESHOLD_1 = THRESHOLD_1, THRESHOLD_2 = THRESHOLD_2, WEIGHT_1 = WEIGHT_1, WEIGHT_2 = WEIGHT_2, WEIGHT_3 = WEIGHT_3)

    hybrid_path = os.path.join(DIR, '../data/hybrid/complete_utility_matrix.csv')
    utility_matrix_complete = pd.read_csv(hybrid_path, index_col=0)

    real_utility_matrix_complete = import_data(matrix_type="real_complete")

    PERFORMANCE_PATH = os.path.join(DIR, '../data/hybrid/performance.txt')
    print('--------------')
    log_to_txt(PERFORMANCE_PATH, '--------------\n')
    # print('Configuration: N_QUERIES =', N_QUERIES, ', N_USERS =', N_USERS, 'THRESHOLD_1 =', THRESHOLD_1, 'THRESHOLD_2 =', THRESHOLD_2)
    # log_to_txt(PERFORMANCE_PATH,'Configuration: N_QUERIES = ' + str(N_QUERIES) + ', N_USERS = ' + str(N_USERS) + ' THRESHOLD_1 = ' + str(THRESHOLD_1) + ' THRESHOLD_2 = ' + str(THRESHOLD_2) + '\n')

    print('Configuration:', 'THRESHOLD_1 =', best_threshold_1, 'THRESHOLD_2 =', best_threshold_2, 'WEIGHT_1 =', best_weight_1, 'WEIGHT_2 =', best_weight_2, 'WEIGHT_3 =', best_weight_3)



    # calculate and printing the performances
    print('\033[1m' + 'Performance of the hybrid algorithm:' + '\033[0m')
    log_to_txt(PERFORMANCE_PATH,'Performance of the hybrid algorithm:\n')

    # mean absolute error: might be helped by the correct prediction of the 0s
    mae = calculate_mae(real_utility_matrix_complete, utility_matrix_complete)
    print('MAE: ', mae)
    log_to_txt(PERFORMANCE_PATH, 'MAE: ' + str(mae) + '\n')

    # RMSE is sensitive to outliers, since the square operation magnifies larger errors.
    rmse = calculate_rmse(real_utility_matrix_complete, utility_matrix_complete)
    print('RMSE :', rmse)
    log_to_txt(PERFORMANCE_PATH, 'RMSE: ' + str(rmse) + '\n')

    mape = calculate_mape(real_utility_matrix_complete, utility_matrix_complete)
    print('MAPE :', mape)
    log_to_txt(PERFORMANCE_PATH, 'MAPE: ' + str(mape) + '\n')

    mre = calculate_mre(real_utility_matrix_complete, utility_matrix_complete)
    print('MRE: ', mre)
    log_to_txt(PERFORMANCE_PATH, 'MRE: ' + str(mre) + '\n')







