import csv
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import OneHotEncoder

from item_item_CF import import_data

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

def import_dataset(path):
    data = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            tuple = []
            for i in range(1, len(row)):
                tuple.append(row[i])
            data.append(tuple)
    return data

def get_users_queries(partial_utility_matrix):
    users_queries_rated = []
    for i in range(len(partial_utility_matrix)):
        user_queries_rated = []
        for j in range(len(partial_utility_matrix.columns)):
            if not np.isnan(partial_utility_matrix.iloc[i, j]):
                user_queries_rated.append((j, partial_utility_matrix.iloc[i, j]))
        users_queries_rated.append(user_queries_rated)

    # sort the queries by rating
    for i in range(len(users_queries_rated)):
        users_queries_rated[i] = sorted(users_queries_rated[i], key=lambda x: x[1], reverse=True)

    return users_queries_rated

# get the number of the TOP_Q queries for each user
def get_top_queries(users_queries_rated, TOP_Q=10):

    users_top_q_queries = []
    for i in range(len(users_queries_rated)):
        user_top_q_queries = []
        TOP_Q_temp = TOP_Q
        if TOP_Q > len(users_queries_rated[i]):
            TOP_Q_temp = len(users_queries_rated[i])
        for j in range(TOP_Q_temp):
            user_top_q_queries.append((queries[users_queries_rated[i][j][0]], users_queries_rated[i][j][1]))
        users_top_q_queries.append(user_top_q_queries)
    return users_top_q_queries

def get_users_profiles(users_top_k_queries):

    # TODO, MISSING: user weight (severity)

    # Strategy 1: using the rating of TOP_Q queries

    # a map where the key is the positional index of the attribute of a query and the value is the list of tuples formed with values of the attribute and the summed rating queries that have that value
    # e.g. {0: [('action', 189), ('drama', 77)], 1: [('france', 88), ('italy', 160), ('mexico', 44)], ...}
    # this map will be done for each user
    users_queries_map = []
    for i in range(len(users_top_k_queries)):
        user_queries_map = {}
        for j in range(len(users_top_k_queries[i])):
            for k in range(len(users_top_k_queries[i][j][0])):
                # skipping the missing values
                if users_top_k_queries[i][j][0][k] == '':
                    continue

                if k not in user_queries_map:
                    user_queries_map[k] = [(users_top_k_queries[i][j][0][k], users_top_k_queries[i][j][1])]
                else:
                    # check if the value is already in the list, if yes, make the mean of the rating
                    value_found = False
                    for l in range(len(user_queries_map[k])):
                        if users_top_k_queries[i][j][0][k] == user_queries_map[k][l][0]:
                            value_found = True
                            user_queries_map[k][l] = (user_queries_map[k][l][0], (user_queries_map[k][l][1] + users_top_k_queries[i][j][1]) / 2)
                            break
                    if not value_found:
                        user_queries_map[k].append((users_top_k_queries[i][j][0][k], users_top_k_queries[i][j][1]))

       # loop the map and sort the values by rating, keeping only one value for each attribute
        #for key in user_queries_map:
            #user_queries_map[key] = sorted(user_queries_map[key], key=lambda x: x[1], reverse=True)
            #user_queries_map[key] = user_queries_map[key][:1]

        # edge case handling: if the map has missing keys, add the missing keys with an empty list
        for key in range(len(queries[0])):
            if key not in user_queries_map:
                user_queries_map[key] = [((''), 0)]

        # sort the map by key
        user_queries_map = dict(sorted(user_queries_map.items()))
        #print('User', i, 'map:', user_queries_map)

        users_queries_map.append(user_queries_map)

    # generate the user profiles as a list
    '''
    users_profiles = []
    for i in range(len(users_queries_map)):
        user_profile = []
        for key in users_queries_map[i]:
            user_profile.append(users_queries_map[i][key][0][0])
        users_profiles.append(user_profile)
    '''

    #return users_profiles, users_queries_map
    return users_queries_map

if __name__ == '__main__':

    TOP_Q = 100

    # query profiles
    queries = import_dataset('../data/_queries.csv')
    print('Number of queries: ', len(queries))
    print('Number of attributes: ', len(queries[0]))
    print('Example of query, query 0: ', queries[0])

    # alternative: movies profiles
    movies = import_dataset('../data/real_data/movies_2.csv')

    ############## CREATING USER PROFILES ##############

    # import partial utility matrix
    partial_utility_matrix = import_data('partial')

    # get users queries rated
    users_queries_rated = get_users_queries(partial_utility_matrix)
    print('Queries of user 9: ', users_queries_rated[9])

    # get the TOP_Q queries for each user
    users_top_k_queries = get_top_queries(users_queries_rated, TOP_Q)
    print('Top ' + str(TOP_Q) + ' queries of user 9: ', users_top_k_queries[9])

    # get the user map of the queries
    users_queries_map = get_users_profiles(users_top_k_queries)



    ###############################


    queries = np.array(queries)


    categories = {}
    # set the keys of the dictionary as the attributes of the queries, values are sets of the possible values of the attributes
    for i in range(len(queries[0])):
        categories[i] = set()

    # add the unique values of each column in the queries dataset to the corresponding set in the categories dictionary
    for i in range(queries.shape[1]):
        for value in np.unique(queries[:, i]):
            categories[i].add(value)

    # remove the empty values from the sets
    for key in categories:
        categories[key].remove('')

    # generate a list of list instead of a dictionary
    categories_list = []
    for key in categories:
        categories_list.append(list(categories[key]))
    print('Categories list: ', categories_list)

    # loop trough each map, for every key, loop trough the categories list and check if the value is in the list, if not, add a tuple with that value and 0, not add duplicates
    for i in range(len(users_queries_map)):
        for key in users_queries_map[i]:
            for j in range(len(categories_list[key])):
                value_found = False
                for k in range(len(users_queries_map[i][key])):
                    if categories_list[key][j] == users_queries_map[i][key][k][0]:
                        value_found = True
                        break
                if not value_found:
                    users_queries_map[i][key].append((categories_list[key][j], 0.0))

    # sort the value alphabetically
    for i in range(len(users_queries_map)):
        for key in users_queries_map[i]:
            users_queries_map[i][key] = sorted(users_queries_map[i][key], key=lambda x: x[0])

    # check if the number of values for each key is the same for all users, should never happen if the code is correct
    for key in users_queries_map[0]:
        for i in range(len(users_queries_map)):
            if i == 0:
                print('First check')
                print('Number of values for key', key, 'is', len(users_queries_map[i][key]))
            if len(users_queries_map[0][key]) != len(users_queries_map[i][key]):
                print('ERROR: Different number of values for key', key, 'for user', i)

    # loop through the users queries map and if the first value of the list of each key is_numeric, make a weighted mean with the values using as weight the second value of the tuple
    for i in range(len(users_queries_map)):
        for key in users_queries_map[i]:
            if is_number(users_queries_map[i][key][0][0]):
                sum = 0
                weight_sum = 0
                for j in range(len(users_queries_map[i][key])):
                    sum += float(users_queries_map[i][key][j][0]) * users_queries_map[i][key][j][1]
                    weight_sum += users_queries_map[i][key][j][1]
                if weight_sum > 0:
                    users_queries_map[i][key] = [(sum / weight_sum, sum / weight_sum)]
                else:
                    users_queries_map[i][key] = [(0, 0)]

    for key in users_queries_map[0]:
        for i in range(len(users_queries_map)):
            if i == 0:
                print('Second check')
                print('Number of values for key', key, 'is', len(users_queries_map[i][key]))
            if len(users_queries_map[0][key]) != len(users_queries_map[i][key]):
                print('ERROR: Different number of values for key', key, 'for user', i)

    # generate a list of queries map like the users_queries_map,loop through the categories list, if the value is present in the query, set the rating to 1, else 0
    queries_map = []
    for i in range(len(queries)):
        query_map = {}
        for key in categories:
            query_map[key] = []
            for j in range(len(categories_list[key])):
                if categories_list[key][j] == queries[i][key]:
                    query_map[key].append((categories_list[key][j], 100.0))
                else:
                    query_map[key].append((categories_list[key][j], 0.0))
        queries_map.append(query_map)

    # sort the value alphabetically
    for i in range(len(queries_map)):
        for key in queries_map[i]:
            queries_map[i][key] = sorted(queries_map[i][key], key=lambda x: x[0])

    # loop through the queries map and if the first value of the list of each key is_numeric, make a weighted mean with the values using as weight the second value of the tuple
    for i in range(len(queries_map)):
        for key in queries_map[i]:
            if is_number(queries_map[i][key][0][0]):
                sum = 0
                weight_sum = 0
                for j in range(len(queries_map[i][key])):
                    sum += float(queries_map[i][key][j][0]) * queries_map[i][key][j][1]
                    weight_sum += queries_map[i][key][j][1]
                if weight_sum > 0:
                    queries_map[i][key] = [(sum / weight_sum, sum / weight_sum)]
                else:
                    queries_map[i][key] = [(0, 0)]

    print(queries_map[12])
    print(queries[12])



    # generate the user profiles as a list, only a list of the values, not a list of list of tuples
    users_profiles = []
    for i in range(len(users_queries_map)):
        user_profile = []
        for key in users_queries_map[i]:
            for j in range(len(users_queries_map[i][key])):
                user_profile.append(users_queries_map[i][key][j][1])
        users_profiles.append(user_profile)

    # generate the query profiles as a list
    queries_profiles = []
    for i in range(len(queries_map)):
        query_profile = []
        for key in queries_map[i]:
            for j in range(len(queries_map[i][key])):
                query_profile.append(queries_map[i][key][j][1])
        queries_profiles.append(query_profile)

    print('User 0 profile: ', users_profiles[0])
    print('Query 0 profile: ', queries_profiles[0])
    print('Query 12 profile: ', queries_profiles[12])

    # normalize the user profiles in the range 0-1
    for i in range(len(users_profiles)):
        max = 0
        for j in range(len(users_profiles[i])):
            if users_profiles[i][j] > max:
                max = users_profiles[i][j]
        if max == 0:
            max = 1
        for j in range(len(users_profiles[i])):
            users_profiles[i][j] = users_profiles[i][j] / max

    # normalize the queries profiles in the range 0-1
    for i in range(len(queries_profiles)):
        max = 0
        for j in range(len(queries_profiles[i])):
            if queries_profiles[i][j] > max:
                max = queries_profiles[i][j]
        if max == 0:
            max = 1
        for j in range(len(queries_profiles[i])):
            queries_profiles[i][j] = queries_profiles[i][j] / max

    print('User 0 profile after normalization: ', users_profiles[0])
    print('Query 0 profile after normalization: ', queries_profiles[0])
    print('Query 12 profile after normalization: ', queries_profiles[12])




    similarities = cosine_similarity(users_profiles, queries_profiles)
    print('Similarity between user 0 and query 0: ', similarities[0][0])
    print('Similarity between user 0 and query 12: ', similarities[0][12])
    #print('Similarities: \n', similarities)

    # print the top 10 similarities of all the users
    #for i in range(len(similarities[:10])):
    #    print('Top 10 similarities for user', i)
    #    print(sorted(similarities[i], reverse=True)[:20])











