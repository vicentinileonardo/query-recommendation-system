movies_2.csv is a real movie dataset, taken from https://www.kaggle.com/datasets/danielgrijalvas/movies.
The original raw file was cleaned up (with the file csv_cleaner.py) and some columns were removed in order to make it compatible with the syntehtic data (relational table)
In addition the score column was normalized to be in the range 0-5 (instead of 0-10).

movies_1.csv in another real dataset (https://www.kaggle.com/datasets/heemalichaudhari/netflix-movies-and-series) cleaned for the purpose
but contains for the genre columns more than one genre per movie while at the moment only one genre per movie is supported.