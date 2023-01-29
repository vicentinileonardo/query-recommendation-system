Instructions on how to run the programs

1. Prerequisites
In order to run the programs, you need to have the following installed:

- Python (programs were tested with Python 3.8)
- Required Python packages:
    - pandas
    - numpy
    - scikit-learn.metrics
    - matplotlib
    - seaborn


2. Running the programs

Computing utility matrices:
    2.1 Compact Item-Item Collaborative Filtering
    The program is run by executing the following command:
    python3 compact_item_item_cf.py or python compact_item_item_cf.py

    2.2 Compact User-User Collaborative Filtering
    The program is run by executing the following command:
    python3 compact_user_user_cf.py or python compact_user_user_cf.py

    2.3 Expanded Item-Item Collaborative Filtering
    The program is run by executing the following command:
    python3 expanded_item_item_cf.py or python expanded_item_item_cf.py

    2.4 Expanded User-User Collaborative Filtering
    The program is run by executing the following command:
    python3 expanded_user_user_cf.py or python expanded_user_user_cf.py

    2.5 Hybrid
    The program is run by executing the following command:
    python3 hybrid.py or python hybrid.py

The following programs make use of the Complete Utility Matrix by importing an existing csv file, pre-computed with Hybrid
and that can be found in  data/hybrid/complete_utility_matrix.csv

PART A:

    2.6 Return the top-k queries that may be of interest to the user u
    The program is run by executing the following command:
    python3 PART_A.py or python PART_A.py

    If no arguments are provided, the program will ask for the user input.
    The program can be run with the following arguments:


PART B

