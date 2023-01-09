import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# set pandas to print the entire dataframe when printing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

utility_matrix = pd.read_csv('../data/_utility_matrix.csv', index_col=0)


utility_matrix = utility_matrix.T
utility_matrix = utility_matrix.iloc[:55, :100]
utility_matrix = utility_matrix.fillna(np.nan)



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
print(utility_matrix)

'''

# order the rows, using row.count to count the most non nan values
utility_matrix = utility_matrix.reindex(utility_matrix.count(axis=1).sort_values(ascending=False).index)
#print(utility_matrix)


# Subtract the mean of each row (query) from the ratings
centered_matrix = utility_matrix.copy()
for row in range(utility_matrix.shape[0]):
    for col in range(utility_matrix.shape[1]):
        if utility_matrix.iloc[row, col] != np.nan:
            centered_matrix.iloc[row, col] = utility_matrix.iloc[row, col] - utility_matrix.iloc[row, :].mean()

centered_matrix = centered_matrix.fillna(0)


def calculate_rating(utility_matrix, counter, user=4, query=0, top_n=2):
    # cosine similarity between rows
    similarities = cosine_similarity(centered_matrix, centered_matrix)

    # selecting the column related to the similarities of the specific query against the others
    similarities = similarities[query, :]
    print('similarities:', similarities)

    # Sort the similarities in descending order
    sorted_similarities = np.argsort(similarities)[::-1]
    print('sorted_sim:', sorted_similarities)

    # Select the top N similar queries, excluding the query itself
    top_n_similarities = sorted_similarities[1:top_n+1]
    print('top similarities: ', top_n_similarities)

    rating = 0
    for i in top_n_similarities:

        # handle nan values, ref. slide 32 of RecSys slide deck
        if np.isnan(utility_matrix.iloc[i, user]):
            rating += similarities[i] * utility_matrix.iloc[i, :].mean()
        else:
            rating += similarities[i] * utility_matrix.iloc[i, user]
    #print(rating)
    print('sum', similarities[top_n_similarities].sum())
    # handle division by zero
    if similarities[top_n_similarities].sum() == 0:
        print('inside exception')
        # option 1: return the mean of the query
        rating = utility_matrix.iloc[query, :].mean()
        # option 2: return the mean of the user
        # rating = utility_matrix.iloc[:, user].mean()
        counter["c_exception"] += 1
    else:
        rating /= similarities[top_n_similarities].sum()
        counter["c_normal"] += 1
    rating = round(rating)
    return rating, counter

# loop all the cell of utility matrix with nan and calculate the rating
complete_utility_matrix = utility_matrix.copy()
ratings_list = []
counter = {"c_exception": 0, "c_normal": 0}
for row in range(utility_matrix.shape[0]):
    for col in range(utility_matrix.shape[1]):
        if np.isnan(utility_matrix.iloc[row, col]):
            print('row: ', row, 'col: ', col)

            rating, counter_exception = calculate_rating(utility_matrix, counter, user=col, query=row, top_n=4)
            complete_utility_matrix.iloc[row, col] = rating
            print('rating: ', rating)
            # append the rating also to a list
            ratings_list.append(rating)

print(ratings_list)
print(utility_matrix)
print(complete_utility_matrix)
print('counter exception:', counter)


with open('../data/viz.html', 'w') as f:
    f.write(utility_matrix.to_html())
    f.write(complete_utility_matrix.to_html())




