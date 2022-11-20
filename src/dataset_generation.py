import csv
import uuid
import random
from utils import get_random_string, get_random_genre, get_random_int, get_random_float, get_random_year, get_random_country

def generate_user_set():
    with open('../data/user_set.csv', 'w') as csvfile:
        fieldnames = ['user_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1000000):
            writer.writerow({'user_id': uuid.uuid4()})
    csvfile.close()

def generate_relationl_table():
    with open('../data/relational_table.csv', 'w') as csvfile:
        fieldnames = ['name', 'genre', 'length', 'year', 'country', 'rating']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(100):
            writer.writerow({'name': get_random_string(random.randint(5, 15)),
                             'genre': get_random_genre(),
                             'length': get_random_int(60, 200),
                              'year': get_random_year(),
                             'country': get_random_country(),
                             'rating': get_random_float(0, 5)}
                            )
    csvfile.close()

if __name__ == '__main__':
    generate_user_set()
    generate_relationl_table()
