import csv
import numpy as np
import pandas as pd
import sys
import os

DIR = os.path.dirname(__file__)

def get_top_k_queries(partial_utility_matrix, computed_utility_matrix, user=0, top_k=10, onlyNotRated=True):
    if computed_utility_matrix.shape != partial_utility_matrix.shape:
        print('The shape of the two matrices are different! Something went wrong!')
        return

    # create a list of tuples (query, rating)
    query_rating_list = []
    for query in range(computed_utility_matrix.shape[1]):
        if onlyNotRated:
            if np.isnan(partial_utility_matrix.iloc[user, query]):
                query_rating_list.append((query, computed_utility_matrix.iloc[user, query]))
        else:
            query_rating_list.append((query, computed_utility_matrix.iloc[user, query]))

    # sort the list of tuples by rating
    query_rating_list.sort(key=lambda x: x[1], reverse=True)

    return query_rating_list[:top_k]

def save_top_k_queries(partial_utility_matrix, utility_matrix_complete, top_k=10, n_users=10):
    path = '../data/compact_item_item_cf/top_' + str(top_k) + '_queries_n_' + str(n_users) + '_users.txt'
    path = os.path.join(DIR, path)

    with open(path, 'w') as f:
        for user in range(n_users):
            top_k_queries = get_top_k_queries(partial_utility_matrix, utility_matrix_complete, user, top_k)

            row_string = 'Top ' + str(top_k) + ' queries for user [' + str(user) + ']: ' + str(top_k_queries) + ' \n'
            f.write(row_string)


if __name__ == "__main__":

    USER = 0
    TOP_K = 5
    ONLY_NOT_RATED = False

    hybrid_path = os.path.join(DIR, '../data/hybrid/complete_utility_matrix.csv')
    utility_matrix_complete = pd.read_csv(hybrid_path, index_col=0)

    partial_utility_matrix_path = os.path.join(DIR, '../data/_utility_matrix.csv')
    partial_utility_matrix = pd.read_csv(partial_utility_matrix_path, index_col=0)

    real_complete_path = os.path.join(DIR, '../data/_utility_matrix_complete.csv')
    real_utility_matrix_complete = pd.read_csv(real_complete_path, index_col=0)

    print('\033[1m' + 'PART A: Top-K queries that may be of interest to the user u' + '\033[0m')

    if len(sys.argv) > 1:

        if sys.argv[1].isdigit() and int(sys.argv[1]) > 0:
            TOP_K = int(sys.argv[1])
        else:
            TOP_K = 5

        if sys.argv[2].isdigit():
            USER = int(sys.argv[2])
        else:
            USER = 0

        if sys.argv[3] == 'True' or sys.argv[3] == 'true' or sys.argv[3] == '1':
            ONLY_NOT_RATED = True
        else:
            ONLY_NOT_RATED = False

        top_k_queries = get_top_k_queries(partial_utility_matrix, utility_matrix_complete, user=USER, top_k=TOP_K,onlyNotRated=ONLY_NOT_RATED)
        print('\n' + '\033[1m' + 'Top ' + str(TOP_K) + ' queries for user [' + str(USER) + ']: ' + '\033[0m' + str(top_k_queries))

        dataset_path = "../data/_queries.csv"

        with open(dataset_path, 'r') as f:
            reader = csv.reader(f)
            header = next(reader)

            counter = 0
            for query in top_k_queries:
                for row in reader:
                    if int(row[0][1:]) == query[0]:
                        print('Query ' + str(counter) + ': ', row, ' with rating: ', query[1])
                        counter += 1
                        break

                # reset the reader to the beginning of the file
                f.seek(0)
                next(reader)
    else:
        print('No parameters passed to the script. Proceeding asking for input...')

        CONTINUE_RUN = True
        while CONTINUE_RUN:
            while True:
                print('Please provide a k value: ')
                while True:
                    try:
                        TOP_K = int(input())
                        if TOP_K <= 0:
                            print('The user number must be greater 0. Please try again: ')
                        else:
                            print('K provided: ', TOP_K)
                            break
                    except:
                        print('The k number must be an integer. Please try again: ')

                print('Please provide a valid user number (from 0 to 2499): ')
                while True:
                    try:
                        user = int(input())
                        if user < 0 or user > 2499:
                            print('The user number must be between 0 and 2499. Please try again: ')
                        else:
                            print('User to analyze: ', user)
                            break
                    except:
                        print('The user number must be an integer. Please try again: ')

                print('Do you want to consider only the queries that the user has not rated yet? (y/n)')
                while True:
                    try:
                        ONLY_NOT_RATED = input()
                        if ONLY_NOT_RATED == 'y':
                            ONLY_NOT_RATED = True
                            break
                        elif ONLY_NOT_RATED == 'n':
                            ONLY_NOT_RATED = False
                            break
                        else:
                            print('Please provide a valid input (y/n): ')
                    except:
                        print('Please provide a valid input (y/n): ')

                top_k_queries = get_top_k_queries(partial_utility_matrix, utility_matrix_complete, user=USER, top_k=TOP_K, onlyNotRated=ONLY_NOT_RATED)
                print('\n' + '\033[1m' + 'Top ' + str(TOP_K) + ' queries for user [' + str(user) + ']: ' + '\033[0m' + str(top_k_queries))

                dataset_path = "../data/_queries.csv"

                with open(dataset_path, 'r') as f:
                    reader = csv.reader(f)
                    header = next(reader)

                    counter = 0
                    for query in top_k_queries:
                        for row in reader:
                            if int(row[0][1:]) == query[0]:
                                print('Query ' + str(counter) + ': ', row, ' with rating: ', query[1])
                                counter += 1
                                break

                        # reset the reader to the beginning of the file
                        f.seek(0)
                        next(reader)
                break

            print('\nDo you want to continue to ask for Top-K queries? (y/n)')
            while True:
                try:
                    CONTINUE_RUN = input()
                    if CONTINUE_RUN == 'y':
                        CONTINUE_RUN = True
                        break
                    elif CONTINUE_RUN == 'n':
                        CONTINUE_RUN = False
                        break
                    else:
                        print('Please provide a valid input (y/n): ')
                except:
                    print('Please provide a valid input (y/n): ')







