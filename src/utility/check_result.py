import sys
import csv

discrete1=60
discrete2=60

runtime_movie=176
runtime_user=124

year_movie=1995
year_user=1993

score_movie=4

avarage_translation=9


runtime=100-(abs(runtime_movie-runtime_user)/3.11)
year=100-(abs(year_movie-year_user)/0.4)
score=(score_movie-2.5)*3

print(str(((discrete1+discrete2+runtime+year+score)/4)+avarage_translation))
