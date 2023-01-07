import string
import random
import os
import shutil
from scipy.stats import truncnorm


def get_random_string(n):
    return ''.join(random.choice(string.ascii_lowercase) for i in range(n))

def get_random_float(a, b):
    return round(random.uniform(a, b), 2)

def get_random_int(a, b):
    return random.randint(a, b)

# get random decade from 1900 to 2022
def sample_from_set(set,n):
    return random.sample(set,n)

def deleteDataContent():
    dir = os.path.dirname(__file__)
    files = os.path.join(dir, '../data/')    
    shutil.rmtree(files)

    os.mkdir(files)
