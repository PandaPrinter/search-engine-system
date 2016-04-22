# search-engine-system
Search engine using java and apache lucene. 

Part1: Applied TF/IDF and vector space similarity metrics to search most relevant document.

For IDF, I use base-2 logarithm to calculate all the IDF value, which won’t affect the ranking results. In general, my algorithm can be divided into five parts:

1. tFNorm() function:
This function pre-computes the 2-norms of all the documents (unnormalized term frequencies) and store the results for future use.

2. tfIdfNorm() function:
This function pre-computes the 2-norms of all the documents based on both TF and IDF factors (unnormalized term frequencies) and store the results for future use. In this function, I also store all the calculated IDF value of each term for future use.  

3. cosineTf() function:
First, this function goes through every term in the query and generate a query norm vector. Then, I extract each term in the query and calculate the cosine value. At last, I use the results from tFNorm() function, cosine value and query norm vector to get the cosine similarity.

4. cosineTfIdf() function:
The same as the cosineTf() function, but the IDF factor and the results from tfIdfNorm() function need to be applied into the calculation to get the cosine similarity.

5. sortResult() function:
Once we get the results from cosineTf() and cosineTfIdf(), we call this function to sort the results based on the value in descending order. At last we extract the first ten results as the output.

Part2

Implemented Authorities/Hubs and PageRank algorithm 

Part3

Implemented KMeans using both Forgy and Random Partition as initialization methods
