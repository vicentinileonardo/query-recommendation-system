Instructions on how to run the programs

1. Prerequisites
In order to run the programs, you need to have the following installed:

- Python (programs were tested with Python 3.8)
- Required Python packages:
    - pandas
    - numpy
    - scikit-learn
    - matplotlib
    - seaborn

If some of the packages are missing, they can be installed using pip:

pip install pandas
pip install numpy
pip install scikit-learn
pip install matplotlib
pip install seaborn

If both Python 2 and 3 are installed on your system, you may need to use pip3 instead of pip:

pip3 install pandas
pip3 install numpy
pip3 install scikit-learn
pip3 install matplotlib
pip3 install seaborn


2. Running the programs

All the programs are located in the "src" folder.

The computed utility matrices are ALREADY stored in csv format, in the folders:
- data/compact_item_item_cf/complete_utility_matrix.csv, for the utility matrix computed with Compact Item-Item CF component
- data/expanded_item_item_cf/complete_utility_matrix.csv, for the utility matrix computed with Expanded Item-Item CF component
- data/hybrid/complete_utility_matrix.csv, for the utility matrix computed with Hybrid solution

Therefore, you can skip the first step (computing the utility matrices) and go directly to the second step (running the recommendation systems).

It is possible though to RECOMPUTE the utility matrices by running the following programs, but not necessary:

Computing utility matrices:
2.1 Compact Item-Item Collaborative Filtering
The program is run by executing the following command:
python3 compact_item_item_cf.py or python compact_item_item_cf.py

2.2 Expanded Item-Item Collaborative Filtering (computationally expensive)
The program is run by executing the following command:
python3 expanded_item_item_cf.py or python expanded_item_item_cf.py

Note:
Some heavy pre-computations are already computed in order to provide results faster (preprocessed_queries.csv and partial_users_items_utility_matrix.csv).
In order to recompute them (not needed), you need change some lines in the code (uncommenting lines 515 and 525).

2.3 Hybrid
The program is run by executing the following command:
python3 hybrid.py or python hybrid.py


PART A:

The following program make use of the Complete Utility Matrix by importing an existing csv file, pre-computed with Hybrid method
and that can be found in  data/hybrid/complete_utility_matrix.csv (which represent the first task of PART A)

2.4 Return the TOP-K queries that may be of interest to the user u
The program is run by executing the following command:

python3 PART_A.py or python PART_A.py

If no arguments are provided, the program will ask for the USER INPUT in the TERMINAL.
The program can be ALSO run with arguments.
Like in the following example:

python3 PART_A.py 5 0 1
or
python PART_A.py 5 0 1

where:
5: is K (the number of top queries to be returned)
0: is user identifier (u)
1: tells the program to return only queries that have not been rated by the user in the partial utility matrix

If you want to ALSO include queries that have been rated by the user in the partial utility matrix,
use 0 as the third argument


PART B

The following program allows to compute a new Complete Utility Matrix having as columns newly posed queries.

The program is run by executing the following command:

python3 PART_B.py or python PART_B.py

The program will ask for the USER INPUT in the TERMINAL.

The first input regards the modalities for posing new queries:
1 for posing the queries in the terminal
2 for posing the queries from a file (the file must of the same shape of file (data/PART_B/queries_PART_B.csv)

Then the prediction will be printed in the file (data/PART_B/rating_hybrid_prediction.csv)

The current file data/PART_B/rating_hybrid_prediction.csv contains the prediction for the queries in the file data/PART_B/queries_PART_B.csv
and it was the file used to compute the evaluation metrics for the report.



NOTE:
PART_B.py requires a heavy pre-computation, that is stored in data/PART_B/support_hybrid_utility_matrix.csv which is
used by the program by default. If you want to recompute it (not necessary), you need to uncomment line 204 in the code.
