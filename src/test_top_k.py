import csv
import numpy as np
import pandas as pd
import sys
import os

DIR = os.path.dirname(__file__)

def get_top_k_queries_all_users(complete_utility_matrix, N_USERS = 2500, top_k=10):

    # users are rows, queries are columns
    # list of sets of query, the list has N_USERS elements, a set has top_k elements max
    top_k_queries = [set() for _ in range(N_USERS)]

    for user in range(N_USERS):
        for query in range(complete_utility_matrix.shape[1]):
            top_k_queries[user].add((query, complete_utility_matrix.iloc[user, query]))
        top_k_queries[user] = sorted(top_k_queries[user], key=lambda x: x[1], reverse=True)[:top_k]

    return top_k_queries


def save_results(TOP_K = 10, N_USERS = 2500):

    hybrid_path = os.path.join(DIR, '../data/hybrid/complete_utility_matrix.csv')
    hybrid_utility_matrix_complete = pd.read_csv(hybrid_path, index_col=0)

    compact_path = os.path.join(DIR, '../data/item_item_cf/complete_utility_matrix.csv')
    compact_utility_matrix_complete = pd.read_csv(compact_path, index_col=0)

    real_complete_path = os.path.join(DIR, '../data/_utility_matrix_complete.csv')
    real_utility_matrix_complete = pd.read_csv(real_complete_path, index_col=0)

    top_k_queries_tuples_hybrid = get_top_k_queries_all_users(hybrid_utility_matrix_complete, N_USERS=N_USERS, top_k=TOP_K)
    top_k_queries_tuples_compact = get_top_k_queries_all_users(compact_utility_matrix_complete, N_USERS=N_USERS, top_k=TOP_K)
    top_k_queries_tuples_real = get_top_k_queries_all_users(real_utility_matrix_complete, N_USERS=N_USERS, top_k=TOP_K)

    # keep only the queries
    top_k_queries_hybrid = [set([x[0] for x in top_k_queries_tuples_hybrid[i]]) for i in range(N_USERS)]
    top_k_queries_compact = [set([x[0] for x in top_k_queries_tuples_compact[i]]) for i in range(N_USERS)]
    top_k_queries_real = [set([x[0] for x in top_k_queries_tuples_real[i]]) for i in range(N_USERS)]

    # save the top k queries for each user
    with open(os.path.join(DIR, '../data/hybrid/top_' + str(TOP_K) + '_queries.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(top_k_queries_hybrid)

    with open(os.path.join(DIR, '../data/item_item_cf/top_' + str(TOP_K) + '_queries.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(top_k_queries_compact)

    with open(os.path.join(DIR, '../data/real_top_' + str(TOP_K) + '_queries.csv'), 'w') as f:
        writer = csv.writer(f)
        writer.writerows(top_k_queries_real)


def compute_jaccard_similarity(set1, set2):
    #print(set1.intersection(set2))
    #print(set1.union(set2))
    return len(set1.intersection(set2)) / len(set1.union(set2))
    #return len(set1 & set2) / len(set1 | set2)

def log_to_txt(path, text):
    with open(path, 'a') as f:
        f.write(text)


if __name__ == "__main__":


    N_USERS = 2500
    TOP_K_VALUES = [1, 2, 3, 4, 5, 10, 15, 20, 30]

    #save_results(TOP_K, N_USERS)
    df = pd.DataFrame(columns=['top_k', 'algorithm_type', 'jaccard_similarity_value'])
    for TOP_K in TOP_K_VALUES:

        # read the top k queries for each user for each csv file and store them lists of sets, each row is a set of queries
        hybrid_path = os.path.join(DIR, '../data/hybrid/top_k_queries/top_' + str(TOP_K) + '_queries.csv')
        hybrid_top_k_queries = pd.read_csv(hybrid_path, header=None).values.tolist()
        hybrid_top_k_queries = [set(x) for x in hybrid_top_k_queries]

        compact_path = os.path.join(DIR, '../data/item_item_cf/top_k_queries/top_' + str(TOP_K) + '_queries.csv')
        compact_top_k_queries = pd.read_csv(compact_path, header=None).values.tolist()
        compact_top_k_queries = [set(x) for x in compact_top_k_queries]

        real_path = os.path.join(DIR, '../data/top_k_queries/real_top_' + str(TOP_K) + '_queries.csv')
        real_top_k_queries = pd.read_csv(real_path, header=None).values.tolist()
        real_top_k_queries = [set(x) for x in real_top_k_queries]

        # compute the jaccard similarity between the top k queries for each user
        jaccard_similarity_hybrid = [compute_jaccard_similarity(hybrid_top_k_queries[i], real_top_k_queries[i]) for i in range(N_USERS)]
        jaccard_similarity_compact = [compute_jaccard_similarity(compact_top_k_queries[i], real_top_k_queries[i]) for i in range(N_USERS)]

        for i in range(N_USERS):
            df = df.append({'user': i, 'top_k': TOP_K, 'algorithm_type': 'compact', 'jaccard_similarity_value': jaccard_similarity_compact[i]}, ignore_index=True)
            df = df.append({'user': i, 'top_k': TOP_K, 'algorithm_type': 'hybrid', 'jaccard_similarity_value': jaccard_similarity_hybrid[i]},ignore_index=True)


    print(df.head())
    # save the dataframe to a csv file
    df.to_csv(os.path.join(DIR, '../data/df_jaccard_similarity.csv'), index=False)


    '''
    LOG_PATH = os.path.join(DIR, '../data/jaccard_top_k.txt')

    print('TOP_K: ', TOP_K)
    log_to_txt(LOG_PATH, 'TOP_K: ' + str(TOP_K) + '\n')

    #print('Jaccard similarity hybrid: ', np.mean(jaccard_similarity_hybrid))
    print('Jaccard similarity compact: ', np.mean(jaccard_similarity_compact))

    #log_to_txt(LOG_PATH, 'Jaccard similarity hybrid: ' + str(np.mean(jaccard_similarity_hybrid)) + '\n')
    log_to_txt(LOG_PATH, 'Jaccard similarity compact: ' + str(np.mean(jaccard_similarity_compact)) + '\n')

    #print('Lowest jaccard similarity hybrid: ', np.min(jaccard_similarity_hybrid))
    print('Lowest jaccard similarity compact: ', np.min(jaccard_similarity_compact))

    #log_to_txt(LOG_PATH, 'Lowest jaccard similarity hybrid: ' + str(np.min(jaccard_similarity_hybrid)) + '\n')
    log_to_txt(LOG_PATH, 'Lowest jaccard similarity compact: ' + str(np.min(jaccard_similarity_compact)) + '\n')

    #print('Highest jaccard similarity hybrid: ', np.max(jaccard_similarity_hybrid))
    print('Highest jaccard similarity compact: ', np.max(jaccard_similarity_compact))

    #log_to_txt(LOG_PATH, 'Highest jaccard similarity hybrid: ' + str(np.max(jaccard_similarity_hybrid)) + '\n')
    log_to_txt(LOG_PATH, 'Highest jaccard similarity compact: ' + str(np.max(jaccard_similarity_compact)) + '\n')
    log_to_txt(LOG_PATH, '-----------------\n')
    '''





















