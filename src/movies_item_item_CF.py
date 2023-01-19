import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import random
import matplotlib.pyplot as plt
import seaborn as sns
import time
import math
import os

# set pandas to print the entire dataframe when printing
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

N_QUERIES = 100
N_USERS = 4
N_ITEMS = 7669

TOP_N = 2
TOP_K_QUERIES = 10
N_USERS_TOP_K_QUERIES = 10

DIR = os.path.dirname(__file__)

def import_data(matrix_type="partial"):
    if matrix_type == "partial":
        path = os.path.join(DIR, '../data/_utility_matrix.csv')
    elif matrix_type == "real_complete":
        path = os.path.join(DIR, '../data/_utility_matrix_complete.csv')
    elif matrix_type == "relational_data":
        path = os.path.join(DIR, '../data/real_data/movies_2.csv')
    elif matrix_type == "preprocess_queries":
        path = os.path.join(DIR, '../data/movies_item_item_cf/preprocessed_queries.csv')
    elif matrix_type == "user_item_weighted":
        path = os.path.join(DIR, '../data/movies_item_item_cf/partial_users_items_utility_matrix.csv')

    else:
        print('Wrong matrix type!')
        return
    utility_matrix = pd.read_csv(path, index_col=0)
    return utility_matrix


def preprocess_query(partial_utility_matrix_df,relational_data_df):
    items=relational_data_df.head(N_ITEMS).index.values
    queries=partial_utility_matrix_df.head(N_QUERIES).columns.values

    output_dictionary={}

    item_count=0

    for item in items:
        output_dictionary[item]=[] #initialize every list associated to an item
        
        for query in queries:
            isPresent=False #is present certain "item" in certain "query"
            query_result_path = os.path.join(DIR, '../data/query_result/'+query+'.csv')
            query_df = pd.read_csv(query_result_path, index_col=0)

            query_results=query_df.head(N_ITEMS).index.values
            
            for query_result in query_results: #for every item that accomplished a query, check if it's equal to the current item of that represent a column
                if item == query_result:
                    isPresent=True
                    break
            if isPresent:
                output_dictionary[item].append(True)
            else:
                output_dictionary[item].append("")

        print("preprocessing item: "+str(item_count))
        item_count+=1

    preprocess_queries_df=pd.DataFrame(data=output_dictionary, index=queries)
    csv_path= os.path.join(DIR, '../data/movies_item_item_cf/preprocessed_queries.csv')
    preprocess_queries_df.to_csv(csv_path)

    preprocess_queries_df = import_data('preprocess_queries')

    return preprocess_queries_df         


#unweighted version
'''
def create_relational_data_utility_matrix(partial_utility_matrix_df,preprocess_data_df):
    columns=preprocess_data_df.head(N_ITEMS).columns.values
    rows=partial_utility_matrix_df.head(N_USERS).index.values
    queries=partial_utility_matrix_df.head(N_QUERIES).columns.values

    output_dictionary={}

    for item in columns:
        output_dictionary[item]=[]

    user_counter=0
    for user in rows:
        for item in columns:
            
            counter=0 #how many times a relational item appears in rated queries results
            partial_score=0 

            for query in queries:
                if not math.isnan(partial_utility_matrix_df.loc[user,query]) and not math.isnan(preprocess_data_df.loc[query,item]):
                        counter+=1
                        partial_score+=partial_utility_matrix_df.loc[user,query] 
            
            if counter != 0:
                final_score=round(partial_score/counter)
                if final_score>100:
                    final_score=100
                elif final_score<0:
                    final_score=0
            else:
                final_score= ""
            output_dictionary[item].append(final_score)
        user_counter+=1
        print("item "+str(user_counter))

    relational_data_utility_matrix=pd.DataFrame(data=output_dictionary, index=rows)

    csv_path= os.path.join(DIR, '../data/movies_item_item_cf/prova_csv_p.csv')
    relational_data_utility_matrix.to_csv(csv_path)

    return relational_data_utility_matrix '''


def create_relational_data_utility_matrix_weighted(partial_utility_matrix_df,preprocess_data_df):
    columns=preprocess_data_df.head(N_ITEMS).columns.values
    rows=partial_utility_matrix_df.head(N_USERS).index.values
    queries=partial_utility_matrix_df.head(N_QUERIES).columns.values

    output_dictionary={}
    for item in columns:
        output_dictionary[item]=[]

    user_counter=0
    results_in_query={}

    for query in queries:
        try:
            results_in_query[query]=int(preprocess_data_df.loc[query].value_counts())
        except:
            results_in_query[query]=0

    for user in rows:
        for item in columns:
            
            partial_score=0 
            total_weight=0 #sum of all the weight that regards the same item
            denominator=0
            weight_list=[] #initialized for every item, it contains in [0] a rating and in [1] its weight

            for query in queries:
                if not math.isnan(partial_utility_matrix_df.loc[user,query]) and not math.isnan(preprocess_data_df.loc[query,item]):
                        current_weight=results_in_query[query]
                        #print(str(item)+ " "+str(preprocess_data_csv.loc[query].value_counts())+" "+str(query))
                        total_weight+=current_weight
                        weight_list.append([partial_utility_matrix_df.loc[user,query],current_weight])

            for el in weight_list:
                partial_score+=(el[0]*(total_weight/el[1]))
                denominator+=(total_weight/el[1])

            if denominator != 0:
                final_score=round(partial_score/denominator)
                if final_score>100:
                    final_score=100
                elif final_score<0:
                    final_score=0
            else:
                final_score= ""
            output_dictionary[item].append(final_score)
        user_counter+=1
        print("user "+str(user_counter)+", computed weighted ratings")

    relational_data_utility_matrix=pd.DataFrame(data=output_dictionary, index=rows)

    csv_path= os.path.join(DIR, '../data/movies_item_item_cf/partial_users_items_utility_matrix.csv')
    relational_data_utility_matrix.to_csv(csv_path)

    relational_data_utility_matrix = import_data('user_item_weighted')
    return relational_data_utility_matrix 


def create_utility_matrix_from_item_matrix(user_item_df,preprocess_data_df,partial_utility_matrix_df):
    rows=partial_utility_matrix_df.head(N_USERS).index.values
    columns=partial_utility_matrix_df.head(N_QUERIES).columns.values
    items=preprocess_data_df.head(N_ITEMS).columns.values

    output_dictionary={}

    for query in columns:
        output_dictionary[query]=[]

    user_counter=0
    for user in rows:
        for query in columns:
            partial_score=0 
            counter=0 #how many times a relational item appears in rated queries results

            if not math.isnan(partial_utility_matrix_df.loc[user,query]):
                final_score=partial_utility_matrix_df.loc[user,query]
            else:
                for item in items:
                    if not math.isnan(preprocess_data_df.loc[query,item]):
                        counter+=1
                        partial_score+=user_item_df.loc[user,item]

                if counter != 0:
                    final_score=round(partial_score/counter,0)
                else:
                    final_score=0

            output_dictionary[query].append(final_score)
        
        user_counter+=1
        print("recostructing user-query utility matrix, n users computed= "+str(user_counter))

    utility_matrix_complete=pd.DataFrame(data=output_dictionary, index=rows)
    path= os.path.join(DIR, '../data/movies_item_item_cf/complete_utility_matrix.csv')
    utility_matrix_complete.to_csv(path)

    return utility_matrix_complete         

def prepare_matrix(matrix, n_items=100, n_users=100):
    matrix = matrix.T
    matrix = matrix.iloc[:n_items, :n_users]
    return matrix
def preprocess_data(utility_matrix, n_items=100, n_users=100):
    utility_matrix = utility_matrix.copy()
    utility_matrix = utility_matrix.T
    utility_matrix = utility_matrix.iloc[:n_items, :n_users]
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

    # maybe, order the rows, using row.count to count the most non nan values
    #utility_matrix = utility_matrix.reindex(utility_matrix.count(axis=1).sort_values(ascending=False).index)

    utility_matrix_before_pp = utility_matrix.copy()

    centered_matrix = utility_matrix.copy()

    # Addressing issue #01 check if row mean == 0 (only 0s and NAs), if that is true, change the 0.0s with something else
    for item in range(utility_matrix.shape[0]):
        if utility_matrix.iloc[item, :].mean() == 0:
            for user in range(utility_matrix.shape[1]):
                if utility_matrix.iloc[item, user] == 0:  # 0s, not NAs
                    #utility_matrix.iloc[item, user] = round((utility_matrix.iloc[:, user].mean()) / 10)
                    # think to do this step directly onto the centered matrix, without touching the original one
                    centered_matrix.iloc[item, user] = round((utility_matrix.iloc[:, user].mean()) / 10)

    # Centered matrix: subtract the mean of each row (item) from the ratings

    for row in range(utility_matrix.shape[0]):
        for col in range(utility_matrix.shape[1]):
            if utility_matrix.iloc[row, col] != np.nan:
                centered_matrix.iloc[row, col] = utility_matrix.iloc[row, col] - utility_matrix.iloc[row, :].mean()


    #Note: merge the above loops into one


    # filling NAs of the centered matrix
    # option 1: fill na with 0s
    #centered_matrix = centered_matrix.fillna(0)

    # option 2: fill na cells with the mean of their row (item)
    centered_matrix = centered_matrix.apply(lambda row: row.fillna(row.mean()), axis=1)

    #MODIFIED
    #if option 2 is choosen, there could be item without any rating
    centered_matrix = centered_matrix.fillna(0)
    #END MODIFIED

    partial_utility_matrix = utility_matrix.copy()

    ### CLUSTERING STEP ###
    '''
    # perform a k-means clustering on the centered matrix
    # option 1: use the euclidean distance
    kmeans = KMeans(n_clusters=5, random_state=0, n_init=10).fit(centered_matrix)
    # option 2: use the cosine similarity
    #kmeans = KMeans(n_clusters=5, random_state=0, metric='cosine').fit(centered_matrix)
    # use fit_predict to get the cluster labels
    cluster_labels = kmeans.fit_predict(centered_matrix)
    centered_matrix['cluster'] = cluster_labels
    grouped_matrix = centered_matrix.groupby('cluster').mean()
    print('centered_matrix \n')
    print(centered_matrix['cluster'])
    print('grouped_matrix \n')
    print(grouped_matrix)
    '''

    return utility_matrix_before_pp, partial_utility_matrix, centered_matrix

def calculate_rating(utility_matrix, centered_matrix, counter, user=4, item=0, top_n=2, verbose=False):
    # cosine similarity between rows
    similarities = cosine_similarity(centered_matrix, centered_matrix)

    # selecting the column related to the similarities of the specific item against the others
    similarities = similarities[item, :]
    if verbose:
        print('Similarities:', similarities)

    # Sort the similarities in descending order
    sorted_similarities = np.argsort(similarities)[::-1]
    if verbose:
        print('Sorted similarities:', sorted_similarities)

    # Select the top N similar items, excluding the item itself
    top_n_similarities = sorted_similarities[1:top_n+1]
    if verbose:
        print('Top N similarities:', top_n_similarities)

    rating = 0
    for i in top_n_similarities:
        # handle nan values, ref. slide 32 of RecSys slide deck, using the mean of the item
        if np.isnan(utility_matrix.iloc[i, user]):
            rating += similarities[i] * utility_matrix.iloc[i, :].mean()
        else:
            rating += similarities[i] * utility_matrix.iloc[i, user]
    # print('sum', similarities[top_n_similarities].sum())

    # handle division by zero, even after the preprocessing, this could happen only if the matrix is very small
    if similarities[top_n_similarities].sum() == 0:
        # print('inside exception')

        # option 1: return the mean of the item
        rating = utility_matrix.iloc[item, :].mean()

        # option 2: return the mean of the user
        # rating = utility_matrix.iloc[:, user].mean()

        # option 3
        # rating = 0

        counter["c_exception"] += 1
    else:  # normal behaviour
        rating /= similarities[top_n_similarities].sum()
        counter["c_normal"] += 1

    #MODIFIED
    if math.isnan(rating):
        rating=0
    #END_MODIFIED

    rating = round(rating)

    if rating > 100:
        rating = 100
    elif rating < 0:
        rating = 0

    return rating, counter

def collaborative_filtering(utility_matrix, centered_matrix, top_n=2, verbose=False):
    # loop all the cell of utility matrix with nan and calculate the rating
    complete_utility_matrix = utility_matrix.copy()
    if verbose:
        print('Complete matrix shape:', complete_utility_matrix.shape)
    ratings_list = []
    counter = {"c_exception": 0, "c_normal": 0}

    for row in range(utility_matrix.shape[0]):
        for col in range(utility_matrix.shape[1]):
            if np.isnan(utility_matrix.iloc[row, col]):
                if verbose:
                    print('Row:', row, 'Col:', col)
                rating, counter_exception = calculate_rating(utility_matrix, centered_matrix, counter, user=col, item=row, top_n=top_n, verbose=verbose)
                complete_utility_matrix.iloc[row, col] = rating
                # append the rating also to a list
                ratings_list.append(rating)

                # rating2, calculated with the clustering

        print('item: ', row, 'Estimated percentage of completion: ', round((row / utility_matrix.shape[0]) * 100, 2), '%')

    if verbose:
        print('Counter: ', counter)

    return complete_utility_matrix, ratings_list

def compute_difference(utility_matrix_complete, n_items=100, n_users=100):

    real_utility_matrix_complete = import_data("real_complete")
    real_utility_matrix_complete = real_utility_matrix_complete.T
    real_utility_matrix_complete = real_utility_matrix_complete.iloc[:n_items, :n_users]

    difference_utility_matrix = (utility_matrix_complete - real_utility_matrix_complete)
    return difference_utility_matrix

# testing performances
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

def print_data_to_html(utility_matrix_before_pp, partial_utility_matrix, centered_matrix, utility_matrix_complete, difference_utility_matrix):
    with open(os.path.join(DIR, '../data/movies_item_item_cf/viz.html'), 'w') as f:
        f.write("<html><head><title>Visualization</title></head><body>")
        f.write("<h1>Utility Matrix Before Preprocessing</h1>")
        f.write(utility_matrix_before_pp.to_html())
        f.write("<h1>Partial Utility Matrix</h1>")
        f.write(partial_utility_matrix.to_html())
        f.write("<h1>Centered Matrix</h1>")
        f.write(centered_matrix.to_html())
        f.write("<h1>Computed Complete Utility Matrix</h1>")
        f.write(utility_matrix_complete.to_html())
        f.write("<h1>Difference Utility Matrix</h1>")
        f.write(difference_utility_matrix.to_html())

# function that plot the heatmap of the utility matrix
def plot_heatmap(utility_matrix, title, annot=True, n_rows=100, n_columns=100):
    utility_matrix = utility_matrix.copy()
    utility_matrix = utility_matrix.iloc[:n_rows, :n_columns]
    plt.figure(figsize=(12,16), dpi=75)
    sns.heatmap(utility_matrix, cmap='coolwarm', center=0, annot=annot)

    # Decorate
    plt.title(title, fontsize=22)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # save the plot
    plt.savefig(os.path.join(DIR, '../data/movies_item_item_cf/heatmap.png'))
    plt.show()

def get_top_k_items(partial_utility_matrix, utility_matrix_complete, user=0, top_k=10):
    if partial_utility_matrix.shape != utility_matrix_complete.shape:
        print('The shape of the two matrices are different! Something went wrong!')
        return

    # create a list of tuples (item, rating)
    item_rating_list = []
    for item in range(partial_utility_matrix.shape[0]):
        # if the item is not rated by the user
        if np.isnan(partial_utility_matrix.iloc[item, user]):
            item_rating_list.append((item, utility_matrix_complete.iloc[item, user]))

    # sort the list of tuples by rating
    item_rating_list.sort(key=lambda x: x[1], reverse=True)

    # return the top N items
    return item_rating_list[:top_k]

def save_top_k_items(partial_utility_matrix, utility_matrix_complete, top_k=10, n_users=10):
    path = '../data/movies_item_item_cf/top_' + str(top_k) + '_items_n_' + str(n_users) + '_users.txt'
    path = os.path.join(DIR, path)
    
    with open(path, 'w') as f:
        for user in range(n_users):
            top_k_items = get_top_k_items(partial_utility_matrix, utility_matrix_complete, user, top_k)

            row_string = 'Top ' + str(top_k) + ' items for user [' + str(user) + ']: ' + str(top_k_items) + ' \n'
            f.write(row_string)

def log_to_txt(path, text):
    with open(path, 'a') as f:
        f.write(text)

def prepare_matrix(matrix, n_items=100, n_users=100):
    matrix = matrix.T
    matrix = matrix.iloc[:n_items, :n_users]
    return matrix

if __name__ == '__main__':

    # load the data
    start_time_load = time.time()
    partial_utility_matrix_df = import_data('partial')
    end_time_load = time.time()
    print('Time taken to load the partial utility matrix:', end_time_load - start_time_load)

    start_time_load = time.time()
    complete_utility_matrix_df = import_data('real_complete')
    end_time_load = time.time()
    print('Time taken to load the complete (real) utility matrix:', end_time_load - start_time_load)

    start_time_load = time.time()
    relational_data_df = import_data('relational_data')
    end_time_load = time.time()
    print('Time taken to load the matrix of relational data:', end_time_load - start_time_load)

    start_time_pp_queries = time.time()
    #preprocess_queries_df = preprocess_query(partial_utility_matrix_df,relational_data_df)
    end_time_pp_queries = time.time()
    print('Time taken to preprocess the queries:', end_time_pp_queries - start_time_pp_queries)

    #uncomment if you want to use a previous precomputation
    start_time_load = time.time()
    preprocess_queries_df = import_data('preprocess_queries')
    end_time_load = time.time()
    print('Time taken to load the matrix of preprocess data:', end_time_load - start_time_load) 
    
    start_time_new_matrix = time.time()
    #new_partial_utility_matrix=create_relational_data_utility_matrix_weighted(partial_utility_matrix_df,preprocess_queries_df)
    end_time_new_matrix = time.time()
    print('Time taken to compute the new user-item utility matrix:', end_time_new_matrix - start_time_new_matrix)

    #uncomment
    start_time_load = time.time()
    new_partial_utility_matrix = import_data('user_item_weighted')
    end_time_load = time.time()
    print('Time taken to load the matrix of user-item-weighted data:', end_time_load - start_time_load) 

    # preprocess the data
    start_time_pp = time.time()
    utility_matrix_before_pp, new_partial_utility_matrix, centered_matrix = preprocess_data(new_partial_utility_matrix, n_items=N_ITEMS, n_users=N_USERS)
    end_time_pp = time.time()
    print('Time taken to preprocess the item partial utility matrix:', end_time_pp - start_time_pp)

    # complete the utility matrix with the ratings
    start_time_cf = time.time()
    items_utility_matrix_complete, ratings_list = collaborative_filtering(new_partial_utility_matrix, centered_matrix, top_n=TOP_N)
    end_time_cf = time.time()
    print('Time taken to fill the item matrix with CF:', end_time_cf - start_time_cf)

    path= os.path.join(DIR, '../data/movies_item_item_cf/users_items_utility_matrix_complete.csv')
    items_utility_matrix_complete.to_csv(path)

    # recostructing from the users-items matrix the users-queries one
    start_time_recostructing = time.time()
    utility_matrix_complete=create_utility_matrix_from_item_matrix(items_utility_matrix_complete.T,preprocess_queries_df,partial_utility_matrix_df)
    utility_matrix_complete=utility_matrix_complete.T
    end_time_recostructing = time.time()
    print('Time taken recostructing the complete users-queries utility matrix:', end_time_recostructing - start_time_recostructing)

    #COMPUTE RESULTS 
    difference_utility_matrix = compute_difference(utility_matrix_complete, n_items=N_ITEMS, n_users=N_USERS)

    # print the data to html, for testing purposes
    print_data_to_html(utility_matrix_before_pp, new_partial_utility_matrix, centered_matrix, utility_matrix_complete, difference_utility_matrix)

    # plot the heatmap of the difference utility matrix
    plot_heatmap(difference_utility_matrix, 'Difference', annot=False, n_rows=N_ITEMS, n_columns=N_USERS)

    real_utility_matrix_complete = import_data('real_complete')
    real_utility_matrix_complete = prepare_matrix(real_utility_matrix_complete, n_items=N_ITEMS, n_users=N_USERS)

    print('--------------')
    log_to_txt(os.path.join(DIR, '../data/movies_item_item_cf/performance.txt'), '--------------\n')
    print('Configuration: N_ITEMS =', N_ITEMS, ', N_USERS =', N_USERS)
    log_to_txt(os.path.join(DIR, '../data/movies_item_item_cf/performance.txt'), 'Configuration: N_ITEMS = ' + str(N_ITEMS) + ', N_USERS = ' + str(N_USERS) + '\n')

    # calculate and printing the performances
    print('\033[1m' + 'Performance of the item-item collaborative filtering algorithm:' + '\033[0m')
    log_to_txt(os.path.join(DIR, '../data/movies_item_item_cf/performance.txt'), 'Performance of the item-item collaborative filtering algorithm:\n')

    # mean absolute error: might be helped by the correct prediction of the 0s
    mae = calculate_mae(real_utility_matrix_complete, utility_matrix_complete)
    print('MAE: ', mae)
    log_to_txt(os.path.join(DIR, '../data/movies_item_item_cf/performance.txt'), 'MAE: ' + str(mae) + '\n')

    # RMSE is sensitive to outliers, since the square operation magnifies larger errors.
    rmse = calculate_rmse(real_utility_matrix_complete, utility_matrix_complete)
    print('RMSE :', rmse)
    log_to_txt(os.path.join(DIR, '../data/movies_item_item_cf/performance.txt'), 'RMSE: ' + str(rmse) + '\n')

    mape = calculate_mape(real_utility_matrix_complete, utility_matrix_complete)
    print('MAPE :', mape)
    log_to_txt(os.path.join(DIR, '../data/movies_item_item_cf/performance.txt'), 'MAPE: ' + str(mape) + '\n')

    mre = calculate_mre(real_utility_matrix_complete, utility_matrix_complete)
    print('MRE: ', mre)
    log_to_txt(os.path.join(DIR, '../data/movies_item_item_cf/performance.txt'), 'MRE: ' + str(mre) + '\n')

    # save the top k items
    save_top_k_items(new_partial_utility_matrix, utility_matrix_complete, top_k=TOP_K_QUERIES, n_users=N_USERS_TOP_K_QUERIES)
