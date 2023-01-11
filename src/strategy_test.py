import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from math import sqrt
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time

# set pandas to print the entire dataframe when printing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# start time
start_time = time.time()

# import the partial utility matrix
utility_matrix = pd.read_csv('../data/_utility_matrix.csv', index_col=0)

utility_matrix = utility_matrix.T
utility_matrix = utility_matrix.iloc[:250, :2500]
utility_matrix = utility_matrix.fillna(np.nan)

# example dataset, from the course slides
'''
dummy_um = pd.DataFrame([[1, np.nan, 3, np.nan, np.nan, 5, np.nan, np.nan, 5, np.nan, 4, np.nan],
                       [np.nan, np.nan, 5, 4, np.nan, np.nan, 4, np.nan, np.nan, 2, 1, 3],
                       [2, 4, np.nan, 1, 2, np.nan, 3, np.nan, 4, 3, 5, np.nan],
                       [np.nan, 2, 4, np.nan, 5, np.nan, np.nan, 4, np.nan, np.nan, 2, np.nan],
                       [np.nan, np.nan, 4, 3, 4, 2, np.nan, np.nan, np.nan, np.nan, 2, 5],
                       [1, np.nan, 3, np.nan, 3, np.nan, np.nan, 2, np.nan, np.nan, 4, np.nan]])

dummy_um.index = utility_matrix.index[:6]
dummy_um.columns = utility_matrix.columns[:12]
utility_matrix = dummy_um
'''

#print("1, utility matrix:\n")
#print(utility_matrix)

# order the rows, using row.count to count the most non nan values
#utility_matrix = utility_matrix.reindex(utility_matrix.count(axis=1).sort_values(ascending=False).index)

#print(utility_matrix)

partial_utility_matrix = utility_matrix.copy()

# Addressing issue #01 check if row mean == 0 (only 0s and NAs), if that is true,  change the 0.0s with 1.0s
for query in range(utility_matrix.shape[0]):
    if utility_matrix.iloc[query, :].mean() == 0:
        for user in range(utility_matrix.shape[1]):
            if utility_matrix.iloc[query, user] == 0:
                utility_matrix.iloc[query, user] = round((1 * utility_matrix.iloc[:, user].mean()) % 10)



# Subtract the mean of each row (query) from the ratings
centered_matrix = utility_matrix.copy()
for row in range(utility_matrix.shape[0]):
    for col in range(utility_matrix.shape[1]):
        if utility_matrix.iloc[row, col] != np.nan:
            centered_matrix.iloc[row, col] = utility_matrix.iloc[row, col] - utility_matrix.iloc[row, :].mean()

#Note: merge the above loops into one

#print("2, centered matrix, before:\n")
#print(centered_matrix)



# filling NAs of the centered matrix
# option 1: fill na with 0s
#centered_matrix = centered_matrix.fillna(0)

# option 2: fill na cells with the mean of their row (query)
centered_matrix = centered_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)


#print("3, centered matrix, after:\n")
#print(centered_matrix)

def calculate_rating(utility_matrix, counter, user=4, query=0, top_n=2):
    # cosine similarity between rows
    similarities = cosine_similarity(centered_matrix, centered_matrix)

    # selecting the column related to the similarities of the specific query against the others
    similarities = similarities[query, :]
    #print('similarities:', similarities)

    # Sort the similarities in descending order
    sorted_similarities = np.argsort(similarities)[::-1]
    #print('sorted_sim:', sorted_similarities)

    # Select the top N similar queries, excluding the query itself
    top_n_similarities = sorted_similarities[1:top_n+1]
    #print('top similarities: ', top_n_similarities)

    rating = 0
    for i in top_n_similarities:
        # handle nan values, ref. slide 32 of RecSys slide deck, using the mean of the query
        if np.isnan(utility_matrix.iloc[i, user]):
            rating += similarities[i] * utility_matrix.iloc[i, :].mean()
        else:
            rating += similarities[i] * utility_matrix.iloc[i, user]
    #print(rating)
    #print('sum', similarities[top_n_similarities].sum())

    # handle division by zero
    if similarities[top_n_similarities].sum() == 0:
        #print('inside exception')

        # option 1: return the mean of the query
        rating = utility_matrix.iloc[query, :].mean()

        # option 2: return the mean of the user
        # rating = utility_matrix.iloc[:, user].mean()

        # option 3
        #rating = 0

        counter["c_exception"] += 1
    else: # normal behaviour
        rating /= similarities[top_n_similarities].sum()
        counter["c_normal"] += 1
    rating = round(rating)

    if rating > 100:
        rating = 100
    elif rating < 0:
        rating = 0

    return rating, counter

'''
# test
counter = {"c_exception": 0, "c_normal": 0}
test_rating, test_counter = calculate_rating(utility_matrix, counter, user=4, query=0, top_n=2)
print('test rating:', test_rating)
print('test counter:', test_counter)

'''

# loop all the cell of utility matrix with nan and calculate the rating
complete_utility_matrix = utility_matrix.copy()
print('complete matrix shape:', complete_utility_matrix.shape)
ratings_list = []
counter = {"c_exception": 0, "c_normal": 0}
for row in range(utility_matrix.shape[0]):
    for col in range(utility_matrix.shape[1]):
        if np.isnan(utility_matrix.iloc[row, col]):
            #print('row: ', row, 'col: ', col)

            rating, counter_exception = calculate_rating(utility_matrix, counter, user=col, query=row, top_n=4)
            complete_utility_matrix.iloc[row, col] = rating
            #print('rating: ', rating)
            # append the rating also to a list
            ratings_list.append(rating)
    print('Row: ', row, 'Estimated percentage of completion: ', round((row / utility_matrix.shape[0]) * 100, 2), '%')


#print('rating_list:', ratings_list)
#print(utility_matrix)
#print(complete_utility_matrix)
#print('counter exception:', counter)

# end time
end_time = time.time()
print('time:', end_time - start_time)


real_utility_matrix_complete = pd.read_csv('../data/_utility_matrix_complete.csv', index_col=0)
real_utility_matrix_complete = real_utility_matrix_complete.T
real_utility_matrix_complete = real_utility_matrix_complete.iloc[:250, :250]

difference_utility_matrix = (real_utility_matrix_complete - complete_utility_matrix)



# testing performances
'''
# calculate the RMSE
rmse = sqrt(mean_squared_error(real_utility_matrix_complete, complete_utility_matrix))
print('RMSE:', rmse)

# calculate the MAE
mae = mean_absolute_error(real_utility_matrix_complete, complete_utility_matrix)
print('MAE:', mae)

# calculate the MAPE
mape = np.mean(np.abs((real_utility_matrix_complete - complete_utility_matrix) / real_utility_matrix_complete)) * 100
print('MAPE:', mape)
'''


with open('../data/viz.html', 'w') as f:
    f.write(partial_utility_matrix.to_html())
    f.write(utility_matrix.to_html())
    f.write(complete_utility_matrix.to_html())
    f.write(difference_utility_matrix.to_html())



def plot_heatmap(matrix, title):
    fig, ax = plt.subplots(figsize=(100, 100))
    sns.heatmap(matrix, cmap='coolwarm', ax=ax)
    ax.set_title(title)
    plt.show()

plot_heatmap(difference_utility_matrix, 'difference utility matrix')


def get_top_k_queries(partial_utility_matrix, utility_matrix, user=0, top_k=10):
    if partial_utility_matrix.shape != utility_matrix.shape:
        print('The shape of the two matrices are different! Something went wrong!')
        return

    # create a list of tuples (query, rating)
    query_rating_list = []
    for query in range(partial_utility_matrix.shape[0]):
        # if the query is not rated by the user
        if np.isnan(partial_utility_matrix.iloc[query, user]):
            query_rating_list.append((query, utility_matrix.iloc[query, user]))

    # sort the list of tuples by rating
    query_rating_list.sort(key=lambda x: x[1], reverse=True)

    # return the top N queries
    return query_rating_list[:top_k]

# test
# get the top 10 queries for user 0
top_k_queries = get_top_k_queries(partial_utility_matrix, complete_utility_matrix, user=0, top_k=10)
print('top 10 queries for user 0:', top_k_queries)

# get the top 10 queries for user 15
top_k_queries = get_top_k_queries(partial_utility_matrix, complete_utility_matrix, user=15, top_k=10)
print('top 10 queries for user 15:', top_k_queries)


def save_top_k_queries(partial_utility_matrix, utility_matrix, top_k=10):
    for user in range(utility_matrix.shape[1]):
        top_k_queries = get_top_k_queries(partial_utility_matrix, complete_utility_matrix, user, top_k)

    path = '../data/top_' + top_k + '_queries.txt'
    with open(path, 'w') as f:
            f.write('top', top_k, ' queries for user [ ' + str(user) + ']: ' + str(top_k_queries) + ' \n')


