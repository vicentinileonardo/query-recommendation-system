import csv
import uuid
import os
import random
import statistics
import numpy as np
from utils import get_random_string, get_random_int, get_random_float, sample_from_set, deleteDataContent

USER_ROWS = 15000
QUERY_ROWS = 500
NORMAL_STANDARDIZED_X=2.17 #to ensure that only the 1.5% of the users have a discrete value preference under (or above) the nearer min/max value to the mean

DIR = os.path.dirname(__file__)
FILENAME_RELATIONAL = os.path.join(DIR, '../data/real_data/movies_2.csv')
FILENAME_USERS = os.path.join(DIR, '../data/_user_set.csv')
FILENAME_USERS_COMPLETE = os.path.join(DIR, '../data/_complete_user_set.csv')
FILENAME_QUERIES = os.path.join(DIR, '../data/_queries.csv')
FILENAME_UTILITY_MATRIX = os.path.join(DIR, '../data/_utility_matrix.csv')
FILENAME_UTILITY_MATRIX_COMPLETE = os.path.join(DIR, '../data/_utility_matrix_complete.csv')
FOLDER_QUERIES_RESULT = os.path.join(DIR, '../data/query_result/')
SCORE_FIELD = "score"



#it analyze the dataset and it returns two dictionaries of sets:
#one containing as key the name of the discrete field's attributes
#the second containing as key the name of the countinuos field's attributes
#as value they have a set of (not-repeated) value encountered
def analyze_relational_table(): 

    with open(FILENAME_RELATIONAL, 'r') as csv_relational:

        csv_reader_relational = csv.reader(csv_relational, delimiter=',')
        relational_line_counter=0
        relational_col_counter=0
        attribute_value ={} # a provisional dictionary containing all the attribute and respective set of values
        relational_fields =[] #containing the keys of the previous dictionary

        for row_relational in csv_reader_relational:
            if relational_line_counter==0: #analyze header of relational table and set up the keys
                for col_name in row_relational:
                    relational_fields.append(col_name)
                    attribute_value[col_name] = set()
            else: #popolate the relational_fields values
                relational_col_counter=0
                for value in row_relational:
                    if value!="":
                        if value.replace('.','',1).isnumeric(): #if value of dataset can be seen as a number, parse it as float
                            value=float(value)
                        attribute_value[relational_fields[relational_col_counter]].add(value)
                    relational_col_counter+=1
            relational_line_counter+=1

        discrete_fields={} #1/2 of output
        countinuos_fields={} #2/2 of output

        for attribute in relational_fields:
            if len(attribute_value[attribute]) < relational_line_counter*0.75: #if the 75% of the values of a certain attribute is unique, we suppose that it can be seen as an id not relevant for the query generation
                is_countinuos_field=True

                for value in attribute_value[attribute]:
                    if not isinstance(value,float): #the considered attribute is discrete
                        is_countinuos_field=False
                        break
                
                if is_countinuos_field: #choose the dictionary to use
                    countinuos_fields[attribute]=attribute_value[attribute]
                else:
                    discrete_fields[attribute]=attribute_value[attribute]

    csv_relational.close()

    return discrete_fields,countinuos_fields 

#it compute, for every countinuos attribute, the min value, the max value and the mean
def analyze_countinuos_attributes(countinuos_fields):
    print("")

    countinuos_fields_min=[]
    countinuos_fields_max=[]
    countinuos_fields_mean=[]

    for attribute in countinuos_fields:
        min_value=min(countinuos_fields[attribute])
        max_value=max(countinuos_fields[attribute])
        mean_value=statistics.mean(countinuos_fields[attribute])

        print("attribute: "+attribute+" - min: "+str(min_value)+" - max: "+str(max_value)+" - mean: "+str(mean_value))

        countinuos_fields_min.append(min_value)
        countinuos_fields_max.append(max_value)
        countinuos_fields_mean.append(mean_value)

    return countinuos_fields_min, countinuos_fields_max, countinuos_fields_mean


#it generate two user set:
#one containing the ids of |USER_ROWS| users, the other one containing the same ids and also some random preferences that those users have on the attributes of the dataset
#the second user set is used to compute the utility matrix
def generate_user_set(discrete_fields,countinuos_fields,countinuos_fields_min, countinuos_fields_max, countinuos_fields_mean):

    with open(FILENAME_USERS, 'w') as csv_user:
        with open(FILENAME_USERS_COMPLETE, 'w') as csv_user_complete:

            fieldnames = ['user_id']

            writer = csv.DictWriter(csv_user, fieldnames=fieldnames) #header user set of only ids
            writer.writeheader()

            fieldnames_complete = ['user_id', 'average_score_translation']
            for attribute in discrete_fields: #for every discrete attribute are taken multiple values, each one with a different user liking value
                fieldnames_complete.append(attribute+"_1")
                fieldnames_complete.append(attribute+"_2")
                fieldnames_complete.append(attribute+"_3")
                fieldnames_complete.append(attribute+"_4")
                fieldnames_complete.append(attribute+"_5")
                fieldnames_complete.append(attribute+"_6")
                fieldnames_complete.append(attribute+"_7")
                fieldnames_complete.append(attribute+"_8")
                fieldnames_complete.append(attribute+"_9")
                fieldnames_complete.append(attribute+"_10")


            for attribute in countinuos_fields: #for every countinuos attribute is taken only a value
                if attribute != SCORE_FIELD:
                    fieldnames_complete.append(attribute)

            writer_complete = csv.DictWriter(csv_user_complete, fieldnames=fieldnames_complete) #header user set with preferences
            writer_complete.writeheader()

            for i in range(USER_ROWS): #write single rows
                user_id = uuid.uuid4()

                complete_row_dict = {'user_id': user_id, 'average_score_translation': get_random_int(-10,10)} #average_score_translation indicate if an user is severe or not during the rating

                for attribute in discrete_fields:
                    sampled_values=sample_from_set(discrete_fields[attribute],10) #the discrete values are sampled
                    complete_row_dict[attribute+"_1"]=sampled_values[0]
                    complete_row_dict[attribute+"_2"]=sampled_values[1]
                    complete_row_dict[attribute+"_3"]=sampled_values[2]
                    complete_row_dict[attribute+"_4"]=sampled_values[3]
                    complete_row_dict[attribute+"_5"]=sampled_values[4]
                    complete_row_dict[attribute+"_6"]=sampled_values[5]
                    complete_row_dict[attribute+"_7"]=sampled_values[6]
                    complete_row_dict[attribute+"_8"]=sampled_values[7]
                    complete_row_dict[attribute+"_9"]=sampled_values[8]
                    complete_row_dict[attribute+"_10"]=sampled_values[9]

                
                countinuos_attribute_counter=0
                for attribute in countinuos_fields: #the countinuos one are taken to a gaussian with mean the one that appears for the values of the dataset
                    if attribute != SCORE_FIELD:
                        mean_value=countinuos_fields_mean[countinuos_attribute_counter]
                        min_value=countinuos_fields_min[countinuos_attribute_counter]
                        max_value=countinuos_fields_max[countinuos_attribute_counter]

                        if(mean_value-min_value < max_value-mean_value):
                            diffAdmitted=mean_value-min_value
                        else:
                            diffAdmitted=max_value-mean_value

                        sdv=diffAdmitted/NORMAL_STANDARDIZED_X

                        countinuos_attribute_counter+=1

                        complete_row_dict[attribute]=round(np.random.normal(mean_value,sdv,1)[0]) #remove "round" to be more general

                writer.writerow({'user_id': user_id})
                writer_complete.writerow(complete_row_dict)

        csv_user_complete.close()
    csv_user.close()
    return fieldnames_complete

#it generate |FILENAME_QUERIES| by sampling at least one discrete or countinuos attribute from the set of values present in the dataset
def generate_queries(discrete_fields,countinuos_fields):    

    with open(FILENAME_QUERIES, 'w') as csv_queries:
        fieldnames = ['query_id'] #fixed field indipendent from the dataset

        #all the meaningful attribute could be part of the queries
        for attribute in discrete_fields:
            fieldnames.append(attribute)
        
        for attribute in countinuos_fields:
            fieldnames.append(attribute)

        writer = csv.DictWriter(csv_queries, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(QUERY_ROWS):
            query_params ={}

            while not query_params:
                for attribute in discrete_fields:
                    if get_random_int(0, 4) == 1: #an attribute is empty for the 20% of the times
                        query_params[attribute]=sample_from_set(discrete_fields[attribute],1)[0]
                
                for attribute in countinuos_fields:
                    if get_random_int(0, 4) == 1: #an attribute is empty for the 20% of the times
                        query_params[attribute]=sample_from_set(countinuos_fields[attribute],1)[0]

            query_params["query_id"]='q'+str(i)
            writer.writerow(query_params)

    csv_queries.close()


#By reading FILENAME_USERS_COMPLETE, FILENAME_QUERIES and FILENAME_RELATIONAL, write the result of each query (in FOLDER_QUERIES_RESULT) and the utility matrix (in FILENAME_UTILITY_MATRIX and in FILENAME_UTILITY_MATRIX_COMPLETE)
def generate_utility_matrix(discrete_fields,countinuos_fields,fieldnames_user_complete):

    all_fieldnames_relational=[]
    all_field_relational ={}

    all_fieldnames_query = [] #will contain the fieldnames that charactherized a query
    query_params = {} #will contain the value requested by a query. The keys used belongs to all_fieldnames_query


    #STEP 1.1: it's popolated the list all_fieldnames_relational with the names of all the attributes the characterize an item of the relational table
    with open(FILENAME_RELATIONAL, 'r') as csv_relational:
        csv_reader_relational = csv.reader(csv_relational, delimiter=',')
        
        for row_relational in csv_reader_relational: #we want also to consider also not relevant attribute neither present in  discrete_fields and countinuos_field
            for col_name in row_relational:
                all_fieldnames_relational.append(col_name)
            break
    csv_relational.close()

    #STEP 1.2: it's popolated the list all_fieldnames_query with the names of all the attributes the could characterize a query
    for fieldname in discrete_fields:
        all_fieldnames_query.append(fieldname)
    
    for fieldname in countinuos_fields:
        all_fieldnames_query.append(fieldname)
    
    #STEP 2: each query is executed and the result is stored in the file query_id.csv
    with open(FILENAME_QUERIES, 'r') as csv_queries:

        row_count_queries = 0
        column_count_queries = 0
        csv_reader_queries = csv.reader(csv_queries, delimiter=',')

        for row_queries in csv_reader_queries:
            #STEP 2.1: the query params are retrieved
            if row_count_queries != 0: #header is skipped
                column_count_queries=0
                for value in row_queries:
                    if column_count_queries == 0: #it's copied the query_id, usefull to give the name to the file containing the results
                        query_id=value
                        filename_query_result = os.path.join(FOLDER_QUERIES_RESULT,query_id+'.csv')
                    else:
                        query_params[all_fieldnames_query[column_count_queries-1]]=value #it is populated query_params, by retrieving the keys from all_fieldnames_query (the -1 is used because column_count_queries starts from 1)
                    column_count_queries+=1

                #STEP 2.2: the query is simulated on the dataset
                with open(FILENAME_RELATIONAL, 'r') as csv_relational, open(filename_query_result, 'w') as csv_query_result:
                    csv_reader_relational = csv.reader(csv_relational, delimiter=',')

                    row_count_relational = 0

                    writer_query_result = csv.DictWriter(csv_query_result, fieldnames=all_fieldnames_relational)
                    writer_query_result.writeheader()

                    #item-query comparison is done for every relational row
                    for row_relational in csv_reader_relational:
                        if row_count_relational != 0: #the header is skipped
                            value_counter = 0
                            for value in row_relational: #the attributes' values of each row is stored in all_field_relational (that will be compared with query_params)
                                all_field_relational[all_fieldnames_relational[value_counter]]=value
                                value_counter+=1

                            insertRelationalRow=True
                            for fieldname in all_fieldnames_query:
                                #an item satisfy a query if, foreach attribute, it has a missung value OR it has the requested value OR it has the requested value in another format (in INT and not in FLOAT)
                                if not (query_params[fieldname]=="" or query_params[fieldname]==all_field_relational[fieldname] or query_params[fieldname]==all_field_relational[fieldname]+".0"):
                                    insertRelationalRow=False
                                    break
                            if insertRelationalRow:
                                writer_query_result.writerow(all_field_relational) 
            
                        row_count_relational += 1

                    csv_relational.close() #to read from the beginning the file in the next iteration
                    csv_query_result.close()

                    row_count_relational=0 
            row_count_queries+=1
        csv_queries.close() 
    

    #STEP 3: compute the utility matrix by comparing the result of each query (the columns of the matrix) with the preferences of each user (the rows of the matrix)
    with open(FILENAME_UTILITY_MATRIX, 'w') as csv_utility_matrix, open(FILENAME_UTILITY_MATRIX_COMPLETE, 'w') as csv_utility_matrix_complete, open(FILENAME_USERS_COMPLETE, 'r') as csvfile_users:
        fieldnames_utility_matrix = ['user_id']

        row_count_queries = 0

        #header of the utility matrix is composed by user_id + the id of all the queries
        with open(FILENAME_QUERIES, 'r') as csv_queries:
            csv_reader_queries = csv.reader(csv_queries, delimiter=',')
            for row_queries in csv_reader_queries: 
                if row_count_queries != 0: 
                    fieldnames_utility_matrix.append(row_queries[0])
                row_count_queries+=1
        csv_queries.close()

        #the header is written
        writer_utility_matrix = csv.DictWriter(csv_utility_matrix, fieldnames=fieldnames_utility_matrix)
        writer_utility_matrix.writeheader()
        writer_utility_matrix_complete = csv.DictWriter(csv_utility_matrix_complete, fieldnames=fieldnames_utility_matrix)
        writer_utility_matrix_complete.writeheader()

        row_count_users = 0
        row_count_queries = 0
        row_count_query_result = 0
        csv_reader_users = csv.reader(csvfile_users, delimiter=',')

        #the utility matrix is written by rows, and so by users
        for row_users in csv_reader_users:
            utility_matrix_complete_row={}
            utility_matrix_row={}

            if row_count_users != 0:
                if row_count_users%(USER_ROWS/100) == 0:
                    print("Progress in generating utility matrix - "+str(row_count_users*100/USER_ROWS)+"%")

                utility_matrix_complete_row["user_id"]=row_users[0] #added user id in the complete utility matrix, user_id is always fixed in first position
                utility_matrix_row["user_id"]=row_users[0] #added user id in utility matrix

                #are read all the queries
                with open(FILENAME_QUERIES, 'r') as csv_queries:
                    csv_reader_queries = csv.reader(csv_queries, delimiter=',')
                    
                    row_count_queries=0
                    for row_queries in csv_reader_queries: #given an user, foreach query compute a preference score
                        if row_count_queries != 0: #is skipped the header of the query definition file

                            partial_score=0
                            final_score=0

                            relational_row_considered=0 #to compute final score correctly
                            row_count_query_result=0 #the row of the query result file

                            filename_query_result = os.path.join(FOLDER_QUERIES_RESULT,row_queries[0]+'.csv')

                            with open(filename_query_result, 'r') as csv_query_result:
                                csv_reader_query_result = csv.reader(csv_query_result, delimiter=',')

                                for row_query_result in csv_reader_query_result: #for every result of the query
                                    if row_count_query_result != 0: #header is skipped
                                        relational_row_considered += 1
                                        
                                        value_counter=0 #number of parameters of the current query that generated qXX.txt, useful to compute final_score

                                        for value_relational in row_query_result:
                                            toConsider=False

                                            field = all_fieldnames_relational[value_counter] #field name that will be searched in fil qXX.csv
                                            try:
                                                index_user_complete=fieldnames_user_complete.index(field+"_1") #check if "field" refers to a discrete attribute
                                                discreteField=True
                                                toConsider=True
                                            except:
                                                pass

                                            try:
                                                index_user_complete=fieldnames_user_complete.index(field) #check if "field" refers to a countinuos attribute
                                                discreteField=False
                                                toConsider=True
                                            except:
                                                pass

                                            if value_relational == "":
                                                toConsider=False

                                            if toConsider: #if it's neither discrete and countinuos, it's not an attibute considerated
                                                if discreteField:
                                                    if value_relational == row_users[index_user_complete]:
                                                        partial_score+=100
                                                    elif value_relational == row_users[index_user_complete+1]:
                                                        partial_score+=90
                                                    elif value_relational == row_users[index_user_complete+2]:
                                                        partial_score+=80
                                                    elif value_relational == row_users[index_user_complete+3]:
                                                        partial_score+=70
                                                    elif value_relational == row_users[index_user_complete+4]:
                                                        partial_score+=50
                                                    elif value_relational == row_users[index_user_complete+5]:
                                                        partial_score+=40
                                                    elif value_relational == row_users[index_user_complete+6]:
                                                        partial_score+=30
                                                    elif value_relational == row_users[index_user_complete+7]:
                                                        partial_score+=20
                                                    elif value_relational == row_users[index_user_complete+8]:
                                                        partial_score+=10
                                                    elif value_relational == row_users[index_user_complete+9]:
                                                        partial_score+=0
                                                    else: 
                                                        partial_score+=60
                                                else:
                                                    min_value=min(countinuos_fields[field])
                                                    max_value=max(countinuos_fields[field])

                                                    if field != SCORE_FIELD:
                                                        partial_score+= 100-(abs(float(value_relational)-float(row_users[index_user_complete]))/((max_value-min_value)/100)) #preference about the length, less is the difference and better it is. 135 are the minutes between the longest and the shortest relational_row
                                                    else:
                                                        partial_score+= (float(value_relational)-2.5)*3

                                            value_counter+=1
                                           
                                    row_count_query_result+=1
                                
                                
                                if relational_row_considered != 0:
                                    final_score=partial_score/(relational_row_considered*(len(discrete_fields)+len(countinuos_fields)-1)) # the -1 is added because we are not considering the "score" field in "countinuos_fields"
                                    final_score+=int(row_users[1]) #average score translation is always in the same position
                                    final_score+=random.randint(-5, 5) #partial_score translated in order to give a less predictable/ more noisy result

                                else:
                                    final_score=0 
                                
                                if final_score<=0:
                                    final_score=0

                                if final_score>=100:
                                    final_score=100

                                utility_matrix_complete_row[row_queries[0]]=round(final_score)

                                if get_random_int(1, 3) == 1: #the "partial" utility matrix has empty the 2/3rd of the positions
                                    utility_matrix_row[row_queries[0]]=round(final_score)

                            csv_query_result.close()

                        row_count_queries+=1

                    writer_utility_matrix_complete.writerow(utility_matrix_complete_row)
                    writer_utility_matrix.writerow(utility_matrix_row)

                    csv_queries.close() #to read from the beginning the file in the next iteration

            row_count_users+=1

    csvfile_users.close()
    csv_utility_matrix.close()
    csv_utility_matrix_complete.close() 


if __name__ == '__main__':
    discrete_fields,countinuos_fields=analyze_relational_table()
    print("discrete_fields")
    print(discrete_fields.keys())
    print()
    print("countinuos_fields")
    print(countinuos_fields.keys())
    print()

    countinuos_fields_min, countinuos_fields_max, countinuos_fields_mean = analyze_countinuos_attributes(countinuos_fields)

    fieldnames_user_complete=generate_user_set(discrete_fields,countinuos_fields,countinuos_fields_min, countinuos_fields_max, countinuos_fields_mean)
    print()
    print("User fieldnames proposed")
    print(fieldnames_user_complete)
    print()

    generate_queries(discrete_fields,countinuos_fields)
    generate_utility_matrix(discrete_fields,countinuos_fields,fieldnames_user_complete)

