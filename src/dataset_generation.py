import csv
import uuid
import random
import os
from utils import get_random_string, get_random_genre, get_random_int, get_random_float, get_random_year,  get_random_length, get_random_country, deleteDataContent

RELATIONAL_ROWS = 1000
USER_ROWS = 10000
QUERY_ROWS = 250


def generate_user_set_old():
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../data/_user_set.csv')

    with open(filename, 'w') as csvfile:
        fieldnames = ['user_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(100):
            writer.writerow({'user_id': uuid.uuid4()})
    csvfile.close()

def generate_user_set():
    dir = os.path.dirname(__file__)
    filename = os.path.join(dir, '../data/_user_set.csv')
    filename_complete = os.path.join(dir, '../data/_complete_user_set.csv')

    with open(filename, 'w') as csvfile:
        with open(filename_complete, 'w') as csvfile_complete:

            fieldnames = ['user_id']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            fieldnames_complete = ['user_id', 'preferred_genre_1', 'preferred_genre_2', 'less_preferred_genre', 'preferred_length', 'preferred_year', 'preferred_country_1', 'preferred_country_2', 'less_preferred_country', 'avarage_rating_translation']
            writer_complete = csv.DictWriter(csvfile_complete, fieldnames=fieldnames_complete)
            writer_complete.writeheader()

            for i in range(USER_ROWS):
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
    filename = os.path.join(dir, '../data/_relational_table.csv')

    with open(filename, 'w') as csvfile:
        fieldnames = ['name', 'genre', 'length', 'year', 'country', 'rating']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(RELATIONAL_ROWS):
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
    filename = os.path.join(dir, '../data/_queries.csv')
    
    counter=0

    with open(filename, 'w') as csvfile:
        fieldnames = ['query_id','genre', 'length', 'year', 'country']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i in range(QUERY_ROWS):
            genre = ""
            length = ""
            year = ""
            country = ""
            rating = ""
            emptyQuery = True

            if get_random_int(0, 1) == 1:
                genre = get_random_genre(1)[0]
                emptyQuery = False
            
            if get_random_int(0, 1) == 1:
                length = get_random_length(1)[0]
                emptyQuery = False

            if get_random_int(0, 1) == 1:
                year = get_random_year(1)[0]
                emptyQuery = False

            if get_random_int(0, 1) == 1 or emptyQuery:
                country = get_random_country(1)[0]

            writer.writerow({'query_id': 'q'+str(counter),
                            'genre': genre,
                            'length': length,
                            'year': year,
                            'country': country}
                            )
            counter+=1

    csvfile.close()

#function that generates an utility matrix, it needs to change the weights given to the attributes and also to take into account the "rating" field of the movies
def generate_utility_matrix():
    dir = os.path.dirname(__file__)
    filename_utility_matrix = os.path.join(dir, '../data/_utility_matrix.csv')
    filename_utility_matrix_complete = os.path.join(dir, '../data/_utility_matrix_complete.csv')
    filename_queries = os.path.join(dir, '../data/_queries.csv')
    filename_users = os.path.join(dir, '../data/_complete_user_set.csv')
    filename_movies = os.path.join(dir, '../data/_relational_table.csv')

    fieldnames_movies = ['name', 'genre', 'length', 'year', 'country', 'rating']

    with open(filename_queries, 'r') as csvfile_queries:

        line_count_queries = 0
        csv_reader_queries = csv.reader(csvfile_queries, delimiter=',')

        #from here creation of files with the result of a query
        for row_queries in csv_reader_queries:
            if line_count_queries != 0:
                filename_query_result = os.path.join(dir, '../data/'+row_queries[0]+'.csv')
                genre=row_queries[1]
                length=row_queries[2]
                year=row_queries[3]
                country=row_queries[4]

                with open(filename_movies, 'r') as csvfile_movies, open(filename_query_result, 'w') as csvfile_query_result:
                    line_count_movies = 0
                    csv_reader_movies = csv.reader(csvfile_movies, delimiter=',')

                    writer_query_result = csv.DictWriter(csvfile_query_result, fieldnames=fieldnames_movies)
                    writer_query_result.writeheader()

                    for row_movies in csv_reader_movies:
                        if line_count_movies != 0:
                            if (genre==row_movies[1] or genre=="") and (length==row_movies[2] or length=="") and (year==row_movies[3] or year=="") and (country==row_movies[4] or country==""):
                                writer_query_result.writerow({'name': row_movies[0],
                                                            'genre': row_movies[1],
                                                            'length': row_movies[2],
                                                            'year': row_movies[3],
                                                            'country': row_movies[4],
                                                            'rating': row_movies[5]}
                                                            )
                        line_count_movies += 1
                    csvfile_movies.close() #to read from the beginning the file in the next iteration
                    line_count_movies=0
            line_count_queries+=1
        csvfile_queries.close()
        
    #from here score computation
    with open(filename_utility_matrix, 'w') as csvfile_utility_matrix, open(filename_utility_matrix_complete, 'w') as csvfile_utility_matrix_complete, open(filename_users, 'r') as csvfile_users:
        fieldnames = ['user_id']

        line_count_queries = 0

        #retrieve queries id for header
        with open(filename_queries, 'r') as csvfile_queries:
            csv_reader_queries = csv.reader(csvfile_queries, delimiter=',')
            for row_queries in csv_reader_queries: 
                if line_count_queries != 0: 
                    fieldnames.append(row_queries[0])
                line_count_queries+=1
        csvfile_queries.close()
        

        writer = csv.DictWriter(csvfile_utility_matrix, fieldnames=fieldnames)
        writer.writeheader()
        writer_complete = csv.DictWriter(csvfile_utility_matrix_complete, fieldnames=fieldnames)
        writer_complete.writeheader()

        line_count_users = 0
        line_count_queries = 0
        line_count_query_result = 0
        csv_reader_users = csv.reader(csvfile_users, delimiter=',')

        for row_users in csv_reader_users:
            utility_matrix_complete_row={}
            utility_matrix_row={}

            if line_count_users != 0:
                utility_matrix_complete_row["user_id"]=row_users[0] #added user id in utility matrix
                utility_matrix_row["user_id"]=row_users[0] #added user id in utility matrix

                with open(filename_queries, 'r') as csvfile_queries:
                    csv_reader_queries = csv.reader(csvfile_queries, delimiter=',')
                    
                    line_count_queries=0
                    for row_queries in csv_reader_queries: #given an user, foreach query compute a preference core
                        if line_count_queries != 0:

                            partial_score=0
                            movie_considered=0
                            final_score=0
                            line_count_query_result=0

                            filename_query_result = os.path.join(dir, '../data/'+row_queries[0]+'.csv')

                            with open(filename_query_result, 'r') as csvfile_query_result:
                                csv_reader_query_result = csv.reader(csvfile_query_result, delimiter=',')

                                for row_query_result in csv_reader_query_result:
                                    if line_count_query_result != 0:
                                        movie_considered+=1
                                        
                                        if row_users[1] == row_query_result[1]: #if the query has the genre that the user loves the most
                                            partial_score+=40
                                        if row_users[2] == row_query_result[1]: #if the query has a genre that the user likes
                                            partial_score+=20
                                        if row_users[3] != row_query_result[1]: #if the query has not the genre that the user hates
                                            partial_score+=60      
                                        
                                        partial_score+= 100-(abs(int(row_query_result[2])-int(row_users[4]))/1.35) #preference about the length, less is the difference and better it is. 135 are the minutes between the longest and the shortest movie
                                        
                                        partial_score+= 100-(abs(int(row_query_result[3])-int(row_users[5]))/1.2) #preference about the length, less is the difference and better it is. 120 are the year between the oldest and the newest movie
                                    
                                        if row_users[6] == row_query_result[4]: #if the query has the nationality that the user loves the most
                                            partial_score+=40
                                        if row_users[7] == row_query_result[4]: #if the query has a nationality that the user likes
                                            partial_score+=20
                                        if row_users[8] != row_query_result[4]: #if the query has not the nationality that the user hates, otherwise the final partial_score for the genre will be 0 for this field
                                            partial_score+=60
                                        
                                        partial_score+=int(row_users[9]) #score translated base on individual avarage translation of score in comparison to the others users (to say: is the user very severe or likes almost everything?)
                                        partial_score+=(int(row_query_result[5])-2.5)*3 #film rating taken into accout 
                                        #putting here partial_score=partial_score/4 is wrong because first partial is divided by 4 for every query result
                                        partial_score+=random.randint(-15, 15) #partial_score translated in order to give a less predictable/ more noisy result
                                    line_count_query_result+=1
                                
                                if movie_considered != 0:
                                    final_score=partial_score/(movie_considered*4) #4 are the attributes that can give a 0 to 100 contribution (genre, length, year, nationality)
                                else:
                                    final_score=0 

                                if final_score<=0:
                                    final_score=0

                                if final_score>=100:
                                    final_score=100  

                                utility_matrix_complete_row[row_queries[0]]=round(final_score)

                                if get_random_int(1, 3) == 1: #the "partial" utility matrix has empty the 2/3rd of the positions
                                    utility_matrix_row[row_queries[0]]=round(final_score)

                            csvfile_query_result.close()

                        line_count_queries+=1

                    writer_complete.writerow(utility_matrix_complete_row)
                    writer.writerow(utility_matrix_row)

                    csvfile_queries.close() #to read from the beginning the file in the next iteration
            line_count_users+=1

    csvfile_users.close()
    csvfile_utility_matrix.close()
    csvfile_utility_matrix_complete.close()


if __name__ == '__main__':
    deleteDataContent()
    generate_user_set()
    generate_relational_table()
    generate_queries()
    generate_utility_matrix()

