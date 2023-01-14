# InformationRetrievalProject
This is a repository for InformationRetrievalProject of the Information Retrieval course at the [Ben-Gurion University](https://in.bgu.ac.il/), Israel.

## Project Description
In this project, we built a search engine for English Wikipedia

## Code
Our project contains the following files: 

● *search_frontend.py*: Flask app for search engine frontend. 
It has six methods as required, including our most important method *search* which contains our implementation for 
the way that the compute engine look for the relevant documents to return to each query.

in our **search** method we firstly tokenize the query in order to remove redundant words, then we do query expansion
by adding similar word using word2vec. now, with the new query, we use BM25 algorithm for the title and for the body of the article, 
each with the relevant inverted index. then, we merge the results from the title and from the body (using the ideal weights). 
after that, we use the pagerank score of each document in order to add to the search score to document that has high pagerank score.
Finally, we send the list of docs with the highest scores for each query. 

● *run_frontend_in_colab.ipynb*: notebook showing how to run the search engine's frontend
in Colab for development purposes. The notebook also provides instructions for
querying/testing the engine.

● *run_frontend_in_gcp.sh*: command-line instructions for deploying the search engine to
GCP.

● *startup_script_gcp.sh*: a shell script that sets up the Compute Engine instance. 

● *create_data_gcp.ipynb*: notebook for creating our inverted indexes in gcp. 
The code in this notebook is creating all of our data, which means it creates the titles inverted index, body inverted index, pagerank dictionary, pageviews dictionary and more.
Because of the large sizes, it creates all the data in google storage bucket, divided to directories, and in every directory it has the relevant bins and pickles files of the posting lists.

● *inverted_index_gcp.py*: this class implements all the methods required for using the inverted indexes. the search_frontend use the methods in this class in order to read and use the inverted index
that was created before. 

● *BM25_from_index.py*: this class implements the relevant methods for using the BM25 algorithm for our compute engine, by using the inverted indexes we created before.
this class implements the BM25 formula we saw in class.

● *cosine_similarity.py*: this class implements the relevant methods for using the cosine similarity algorithm for our compute engine, by using the inverted indexes we created before.
this class implements the efficient cosine similarity calculation we saw in class.

● *tools.py*: this class implements our helper methods that we used for implementing the search functions of our engine (tokenizing function etc.)

