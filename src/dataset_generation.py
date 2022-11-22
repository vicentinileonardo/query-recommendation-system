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

if __name__ == '__main__':
    generate_user_set()
    generate_relational_table()
    generate_queries()

