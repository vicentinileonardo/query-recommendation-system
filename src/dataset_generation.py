# generate csv file for user dataset, only one user_id per line

import csv
import random

def generate_user_set():
    with open('../data/user_set.csv', 'w') as csvfile:
        fieldnames = ['user_id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for i in range(1000000):
            writer.writerow({'user_id': random.randint(1, 1000000)})
    csvfile.close()

if __name__ == '__main__':
    generate_user_set()
