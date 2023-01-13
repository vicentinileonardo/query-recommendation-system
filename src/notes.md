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

## Issue #03
+ MRE is not working properly