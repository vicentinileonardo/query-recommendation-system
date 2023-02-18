import pandas as pd
import csv
import math
import os


DIR = os.path.dirname(__file__)
N_QUERIES = 100
N_USERS = 2500
N_ITEMS = 7669


# trovare quali item soddisfano una query

# per ogni utente prendere le valutazioni di quegli item (matrice completa, abbiamo tutto) e fare la media
def compute_user_item_utility_matrix(complete_utility_matrix,preprocess_queries):
    columns=preprocess_queries.head(N_ITEMS).columns.values
    rows=complete_utility_matrix.head(N_USERS).index.values
    queries=complete_utility_matrix.head(N_QUERIES).columns.values

    output_dictionary={}
    for item in columns:
        output_dictionary[item]=[]

    user_counter=0
    results_in_query={}

    for query in queries:
        try:
            results_in_query[query]=int(preprocess_queries.loc[query].value_counts())
        except:
            results_in_query[query]=0

    for user in rows:
        #user_mean_ratings=complete_utility_matrix.loc[user,:].mean()

        for item in columns:
            
            partial_score=0 
            total_weight=0 #sum of all the weight that regards the same item
            denominator=0
            weight_list=[] #initialized for every item, it contains in [0] a rating and in [1] its weight

            for query in queries:
                if not math.isnan(complete_utility_matrix.loc[user,query]) and not math.isnan(preprocess_queries.loc[query,item]):
                        current_weight=results_in_query[query]
                        total_weight+=current_weight
                        weight_list.append([complete_utility_matrix.loc[user,query],current_weight])

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
                final_score= 404 #PLACEHOLDER
            output_dictionary[item].append(final_score)
        user_counter+=1
        print("user "+str(user_counter)+", computed weighted ratings")
    
    user_counter=0
    for user in rows:
        item_counter=0
        user_mean_rating=0
        for item in columns:
            if output_dictionary[item][user_counter]!=404:
                user_mean_rating+=output_dictionary[item][user_counter]
                item_counter+=1
        user_mean_rating=round(user_mean_rating/item_counter)

        for item in columns:
            if output_dictionary[item][user_counter]==404:
                output_dictionary[item][user_counter]=user_mean_rating

        user_counter+=1
        

    relational_data_utility_matrix=pd.DataFrame(data=output_dictionary, index=rows)

    csv_path= os.path.join(DIR, '../data/PART_B/support_hybrid_utility_matrix.csv')
    relational_data_utility_matrix.to_csv(csv_path)

    return relational_data_utility_matrix 

def preprocess_proposed_queries(queries,relational_data,header):
    items=relational_data.head(N_ITEMS).index.values
    n_queries=len(queries)

    output_dictionary={}

    item_count=0

    for item in items:
        output_dictionary[item]=[] #initialize every list associated to an item

        for i in range(n_queries):
            query = queries[i]
            field_counter=0

            includeResult = True

            for field in query:

                if not (str(relational_data.loc[item,header[field_counter]]) == field) and not (field == ""):
                    try:
                        if float(relational_data.loc[item,header[field_counter]]) == float(field):
                            pass
                        else:
                            includeResult = False
                            break
                    except:
                        includeResult = False
                        break

                field_counter+=1

            
            if includeResult:
                output_dictionary[item].append("True")
            else:
                output_dictionary[item].append("")

        #print("preprocessing item: "+str(item_count))
        item_count+=1

    index=[]

    for i in range(n_queries):
        index.append("q"+str(i))

    preprocess_queries_df=pd.DataFrame(data=output_dictionary, index=index)
    csv_path= os.path.join(DIR, '../data/PART_B/preprocessed_queries.csv')
    preprocess_queries_df.to_csv(csv_path)

def compute_new_utility_matrix(queries,support_utility_matrix, new_preprocess_queries):
    rows=support_utility_matrix.head(N_USERS).index.values
    items=new_preprocess_queries.head(N_ITEMS).columns.values

    n_queries=len(queries)
    output_dictionary={}

    for i in range(n_queries):
        output_dictionary["q"+str(i)]=[]

    user_counter=0
    query_counter=0

    for user in rows:
        query_counter=0
        for i in range(n_queries):
            partial_score=0 
            counter=0 #how many times a relational item appears in rated queries results
            query_id="q"+str(query_counter)

            for item in items:
                if not math.isnan(new_preprocess_queries.loc[query_id,item]):
                    counter+=1
                    partial_score+=support_utility_matrix.loc[user,item]

            if counter != 0:
                final_score=round(partial_score/counter,0)
            else:
                final_score=0

            output_dictionary[query_id].append(final_score)
            query_counter+=1
        
        user_counter+=1
        if user_counter%100==0:
            print("recostructing user-query utility matrix, n users computed= "+str(user_counter))

    ratings_prediction=pd.DataFrame(data=output_dictionary, index=rows)
    path= os.path.join(DIR, '../data/PART_B/ratings_hybrid_prediction.csv')
    ratings_prediction.to_csv(path)

    print(ratings_prediction)

    return ratings_prediction   

def write_txt(queries):
    with open(os.path.join(DIR, '../data/PART_B/queries.txt'), 'w') as f:
        counter=0
        for query in queries:
            f.write("q"+str(counter)+", "+str(query)+"\n")
            counter+=1

if __name__ == "__main__":
    complete_utility_matrix_path = os.path.join(DIR, '../data/hybrid/complete_utility_matrix.csv')
    preprocess_queries_path = os.path.join(DIR, '../data/expanded_item_item_cf/preprocessed_queries.csv')
    support_utility_matrix_path = os.path.join(DIR, '../data/PART_B/support_hybrid_utility_matrix.csv')
    dataset_path = os.path.join(DIR, '../data/_queries.csv')
    relational_data_path = os.path.join(DIR, '../data/real_data/movies_2.csv')
    new_preprocess_queries_path = os.path.join(DIR, '../data/PART_B/preprocessed_queries.csv')


    complete_utility_matrix = pd.read_csv(complete_utility_matrix_path, index_col=0)
    preprocess_queries = pd.read_csv(preprocess_queries_path, index_col=0)

    #compute_user_item_utility_matrix(complete_utility_matrix,preprocess_queries)
    support_utility_matrix = pd.read_csv(support_utility_matrix_path, index_col=0)

    
    print('\033[1m' + 'Rating of a query in general' + '\033[0m')

    with open(dataset_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
    header = header[1:]

    print('The categories of the current dataset are: ', header)
    print('Example 1: Comedy, Italy, 120, 2000, 4')
    print('Example 2: Drama,,, 1970, 5')
    print('There are 2 modalities: \n1. Entering the query directly in the prompt \n2. Reading the queries to pose from a file')

    modality=0
    while modality != 1 and modality != 2:
        print('Choose the modality (type 1 or 2): ')
        modality = input()

        if modality == '1':
            print('Enter a query composed of maximum '+str(len(header))+' words, making sure that the words are separated by a comma')
            print('If you want to skip a category, use a double comma (,,)')
            query = input()
            query = query.split(',')
            query = [x.strip() for x in query]
            query = query[:5]
            print('You entered the query:', query)

            queries = []
            queries.append(query)
            break
            

        elif modality == '2':
            print('Enter the full path of the file containing the queries')
            path = input()
            path = os.path.join(DIR, path)
            file_found = False
            while not file_found:
                try:
                    queries = []
                    with open(path, 'r') as f:
                        reader = csv.reader(f)
                        next(reader)
                        for row in reader:
                            query = row
                            query = [x.strip() for x in query]
                            query = query[:5]
                            queries.append(query)
                    print('The queries to pose are: ', queries)
                    file_found = True
                    break
                    
                except:
                    print('The file does not exist, please check the full path')
                    print('Enter the full path of the file containing the queries')
                    path = input()
            break
        else:
            print('Invalid modality')
            exit
    
    
    write_txt(queries)

    relational_data=pd.read_csv(relational_data_path, index_col=0)
    preprocess_proposed_queries(queries,relational_data,header)

    new_preprocess_queries = pd.read_csv(new_preprocess_queries_path, index_col=0)
    compute_new_utility_matrix(queries,support_utility_matrix, new_preprocess_queries) 


