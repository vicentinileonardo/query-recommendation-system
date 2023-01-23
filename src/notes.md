## Issue #01
+ problem when the query row has as only 0s as rating availables
+ This issue has been solved 

## Issue #02
+ if the matrix is very small, the case where there is only 1 rating led to the excpetion code path
+ probably not relevant, only in testing phase


for q64, good ratings calculated (if an example to check is needed)

## Clustering
Clustering can be useful as a pre-processing step to improve the performance of the CF algorithm. 
Clustering can be used to group similar items together and then the similarity between items within the same cluster can be considered stronger than the similarity between items in different clusters. 
This can help to reduce the dimensionality of the problem and make the CF algorithm more computationally efficient.

Additionally, it can also be used as post-processing step, 
once the similarity matrix is calculated, 
you could group similar items together in clusters and then recommend the items within the same cluster, 
this will help to improve the diversity of the recommendations.

The usefulness of clustering in item-item CF depends on the specific characteristics of the dataset and the problem you are trying to solve, 
so it's worth to test the performance of the algorithm with and without clustering and see which one performs better.

## content based
+ discrete values encoded (values is rating), continuous values as they are (values are the actual values, like the years or rating): similarities too high
+ discrete values encoded, continuous values encoded: similarities too lows, since the vectors are very long and so different: similarites are very low
+ dynamic profiles, best results. dynamic feature selection 
+ still, the perfomances were not as good as the ones of the collaborative filtering, 
+ the initial idea was to use it as a part of a hybrid system, but the results were not good enough to be used in a hybrid system

## in general, for the report
implementative choices were mainly driven by empirical results, using metrics like RMSE and MAE
the choice to spend more time in the dataset creation strategy was useful since it allowed to make comparisons 
it allowed us think if the result made sense or not, and to understand if the results were good or not
instead, using a random dataset, with ratings rand(0,100),  it would have been difficult to understand if the results were good or not
there is coerence between users

maybe test the final algorithm on a random dataset, and describe the difference between the outcomes

## for the report, running various run with different datasets

the solution as generic as possible and it works with any dataset,


# consider to use the basic item-item CF or content-based as a baseline for the hybrid system

does the algorithm works well with many/few rows and many/few columns?
does the algorithm works well with short/long queries?

show that item-item CF beat user-user, show data

novelty of the approach:
+ expansion of the query
+ maybe optimization of the weights with gradient descent

