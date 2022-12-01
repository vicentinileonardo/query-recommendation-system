import csv
import uuid
import random
import os
from utils import get_random_string, get_random_genre, get_random_int, get_random_float, get_random_year,  get_random_length, get_random_country

def generate_user_set_old():
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../data/user_set.csv')

    with open(filename, 'w') as csvfile:
        fieldnames = ['user_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(100):
            writer.writerow({'user_id': uuid.uuid4()})
    csvfile.close()

def generate_user_set():
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../data/user_set.csv')
    filename_complete = os.path.join(dir, '../data/complete_user_set.csv')

    with open(filename, 'w') as csvfile:
        with open(filename_complete, 'w') as csvfile_complete:

            fieldnames = ['user_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            fieldnames_complete = ['user_id', 'preferred_genre_1', 'preferred_genre_2', 'less_preferred_genre', 'preferred_length', 'preferred_year', 'preferred_country_1', 'preferred_country_2', 'less_preferred_country', 'avarage_rating_translation']
            writer_complete = csv.DictWriter(csvfile_complete, fieldnames=fieldnames_complete)
            writer_complete.writeheader()

            for i in range(100):
                user_id = uuid.uuid4()
                genres = get_random_genre(3)
                countries = get_random_country(3)

                writer.writerow({'user_id': user_id})

                writer_complete.writerow({'user_id': user_id,
                                        'preferred_genre_1': genres[0],
                                        'preferred_genre_2': genres[1],
                                        'less_preferred_genre': genres[2],
                                        'preferred_length': get_random_length(1)[0],
                                        'preferred_year': get_random_year(1)[0],
                                        'preferred_country_1': countries[0],
                                        'preferred_country_2': countries[1],
                                        'less_preferred_country': countries[2],
                                        'avarage_rating_translation': random.randint(-15, 15)
                                        }
                                        )

        csvfile_complete.close()
    csvfile.close()

def generate_relational_table():
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../data/relational_table.csv')

    with open(filename, 'w') as csvfile:
        fieldnames = ['name', 'genre', 'length', 'year', 'country', 'rating']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(100):
            writer.writerow({'name': get_random_string(random.randint(5, 15)),
                             'genre': get_random_genre(1)[0],
                             'length': get_random_length(1)[0],
                              'year': get_random_year(1)[0],
                             'country': get_random_country(1)[0],
                             'rating': get_random_int(0, 5)}
                            )
    csvfile.close()

def generate_queries():
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../data/queries.csv')
    
    counter=0

    with open(filename, 'w') as csvfile:
        fieldnames = ['query_id','genre', 'length', 'year', 'country']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        #Queries with 1 attribute
        genres = get_random_genre(4) 
        lengths=get_random_length(4)
        years=get_random_year(4)
        countries=get_random_country(6)

        for i in range(4):
            writer.writerow({'query_id': 'q'+str(counter),
                            'genre': genres[i]}
                            )
            counter+=1

        for i in range(4):
            writer.writerow({'query_id': 'q'+str(counter),
                            'length': lengths[i]}
                            )
            counter+=1

        for i in range(4):
            writer.writerow({'query_id': 'q'+str(counter),
                            'year': years[i]}
                            )
            counter+=1

        for i in range(6):
            writer.writerow({'query_id': 'q'+str(counter),
                            'country': countries[i]}
                            )
            counter+=1

        #Queries with 2 attributes with genre
        genres = get_random_genre(4) 

        for i in range(4):
            lengths=get_random_length(4)
            years=get_random_year(4)
            countries=get_random_country(6)

            for j in range(4):
                writer.writerow({'query_id': 'q'+str(counter),
                                'genre': genres[i],
                                'length': lengths[j]}
                                )
                counter+=1

            for j in range(4):
                writer.writerow({'query_id': 'q'+str(counter),
                                'genre': genres[i],
                                'year': years[j]}
                                )
                counter+=1

            for j in range(6):
                writer.writerow({'query_id': 'q'+str(counter),
                                'genre': genres[i],
                                'country': countries[j]}
                                )
                counter+=1

        #Queries with 2 attributes with length
        lengths=get_random_length(4)

        for i in range(4):
            years=get_random_year(4)
            countries=get_random_country(6)

            for j in range(4):
                writer.writerow({'query_id': 'q'+str(counter),
                                'length': lengths[i],
                                'year': years[j]}
                                )
                counter+=1

            for j in range(6):
                writer.writerow({'query_id': 'q'+str(counter),
                                'length': lengths[i],
                                'country': countries[j]}
                                )
                counter+=1

        
        #Queries with 2 attributes with year
        years=get_random_year(4)

        for i in range(4):
            countries=get_random_country(6)

            for j in range(6):
                writer.writerow({'query_id': 'q'+str(counter),
                                'year': years[i],
                                'country': countries[j]}
                                )
                counter+=1

    csvfile.close()

#function that generates an utility matrix, it needs to change the weights given to the attributes and also to take into account the "rating" field of the movies
def generate_utility_matrix():
    dir = os.path.dirname(__file__)
    filename_utility_matrix = os.path.join(dir, '../data/utility_matrix.csv')
    filename_utility_matrix_complete = os.path.join(dir, '../data/utility_matrix_complete.csv')
    filename_queries = os.path.join(dir, '../data/queries.csv')
    filename_users = os.path.join(dir, '../data/complete_user_set.csv')
    filename_movies = os.path.join(dir, '../data/relational_table.csv')


    with open(filename_utility_matrix, 'w') as csvfile_utility_matrix, open(filename_utility_matrix_complete, 'w') as csvfile_utility_matrix_complete, open(filename_users, 'r') as csvfile_users:
        fieldnames = ['user_id']

        line_count_queries = 0

        with open(filename_queries, 'r') as csvfile_queries:
            csv_reader_queries = csv.reader(csvfile_queries, delimiter=',')
            for row_queries in csv_reader_queries:
                if line_count_queries != 0:
                    fieldnames.append(row_queries[0])
                line_count_queries+=1
            csvfile_queries.close() #to read from the beginning the file in the next iteration


        writer = csv.DictWriter(csvfile_utility_matrix, fieldnames=fieldnames)
        writer.writeheader()
        writer_complete = csv.DictWriter(csvfile_utility_matrix_complete, fieldnames=fieldnames)
        writer_complete.writeheader()

        line_count_users = 0
        line_count_queries = 0
        csv_reader_users = csv.reader(csvfile_users, delimiter=',')

        for row_users in csv_reader_users:
            utility_matrix_complete_row={}
            utility_matrix_row={}

            if line_count_users != 0:
                utility_matrix_complete_row["user_id"]=row_users[0] #added user id in utility matrix
                utility_matrix_row["user_id"]=row_users[0] #added user id in utility matrix

                with open(filename_queries, 'r') as csvfile_queries:
                    csv_reader_queries = csv.reader(csvfile_queries, delimiter=',')

                    for row_queries in csv_reader_queries: #given an user, foreach query compute a preference score
                        if line_count_queries != 0:
                            used_params={}
                            score=0
                            considered_params=0

                            if row_queries[1] != "":
                                considered_params+=1
                                used_params["row_movies[1]"] = row_queries[1]
                                if row_users[1] == row_queries[1]: #if the query has the genre that the user loves the most
                                    score+=40
                                if row_users[2] == row_queries[1]: #if the query has a genre that the user likes
                                    score+=20
                                if row_users[3] != row_queries[1]: #if the query has not the genre that the user hates
                                    score+=60      

                            if row_queries[2] != "":
                                considered_params+=1
                                used_params["row_movies[2]"] = row_queries[2]
                                score+= 100-(abs(int(row_queries[2])-int(row_users[4]))/1.35) #preference about the length, less is the difference and better it is. 135 are the minutes between the longest and the shortest movie

                            if row_queries[3] != "":
                                considered_params+=1
                                used_params["row_movies[3]"] = row_queries[3]
                                score+= 100-(abs(int(row_queries[3])-int(row_users[5]))/1.2) #preference about the length, less is the difference and better it is. 120 are the year between the oldest and the newest movie

                            if row_queries[4] != "":
                                considered_params+=1
                                used_params["row_movies[4]"] = row_queries[4]
                                if row_users[6] == row_queries[4]: #if the query has the nationality that the user loves the most
                                    score+=40
                                if row_users[7] == row_queries[4]: #if the query has a nationality that the user likes
                                    score+=20
                                if row_users[8] != row_queries[4]: #if the query has not the nationality that the user hates, otherwise the final score for the genre will be 0 for this field
                                    score+=60  

                            ##FROM HERE IS TAKEN INTO ACCOUNT THE "RATING" FIELD OF THE SINGLE MOVIES
                            #rating_sum=0
                            #movies_considered=0
                            #with open(filename_movies, 'r') as csvfile_movies:
                                #query=""
                                #counter_params_inserted=0
                                #for key in used_params:
                                    #if counter_params_inserted != 0:
                                        #query+=" and "
                                    #query += key + "==" + used_params[key]
                                    #counter_params_inserted+=1
                                #print(query)

                                #for row_movies in csvfile_movies:
                                    #if eval(query):
                                        #rating_sum+=(row_movies[5]-3)*5
                                        #movies_considered+=1

                                #csvfile_movies.close()

                            #rating_avarage=rating_sum/movies_considered
                            #score+=rating_avarage

                            score=score/considered_params #to obtain a rating between 0 and 100
                            score+=int(row_users[9]) #score translated base on individual avarage translation of scores in comparison to the others users (to say: is the user very severe or likes almost everything?)

                            score+=random.randint(-15, 15) #score translated in order to give a less predictable/ more noisy result
                            
                            score=round(score,0)/100 #to obtain a score between 0 and 1
                            if score<=0:
                                score=0

                            if score>=1:
                                score=1

                            utility_matrix_complete_row[row_queries[0]]=score

                            if get_random_int(1, 3) == 1: #the "partial" utility matrix has empty the 2/3rd of the positions
                                utility_matrix_row[row_queries[0]]=score

                        line_count_queries+=1

                    writer_complete.writerow(utility_matrix_complete_row)
                    writer.writerow(utility_matrix_row)

                    csvfile_queries.close() #to read from the beginning the file in the next iteration
                    line_count_queries=0
            line_count_users+=1

    csvfile_users.close()
    csvfile_utility_matrix.close()
    csvfile_utility_matrix_complete.close()

if __name__ == '__main__':
    generate_user_set()
    generate_relational_table()
    generate_queries()
    generate_utility_matrix()

