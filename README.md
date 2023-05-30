# 5. Assignment 5 – Self Chosen 
## 5.1 Assignment Description 
This is the self-chosen assignment for the exam in Language Analytics. I have chosen to cluster speeches of United States Presidents who are either Democratic or Republic. This assignment will cluster the speeches using two different methods, to see how the clusters change. All clusters will use the KMeans algorithm, but the number of clusters will be decided by _DBI and Silhouette score_ and the _elbow method_. 

This assignment will create scatterplots of the clusters and tables showing which keywords are in the clusters. This assignment aims to see how speeches given by either a Democratic or Republican president cluster.
## 5.2 Machine Specifications and My Usage
All the computation done for this project was performed on the UCloud interactive HPC system, which is managed by the eScience Center at the University of Southern Denmark. The scripts were created with Coder Python 1.73.1 and Python version 3.9.2. I ran the scripts on 16 CPUs, and a total run time of three minutes. 
### 5.2.1 Perquisites
To run the scripts, make sure to have Bash and Python3 installed on your device. The script has only been tested on Ucloud.
## 5.3 Contribution
The scripts created for this assignment take inspiration from Andrea D’Agostino’s article [Text Clustering with TF-IDF in Python](https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7). This article gives a descriptive guide on how to cluster texts using TF-IDF in Python. Furthermore, this assignment uses data gathered by Kaggle user [Joseph Lilleberg](https://www.kaggle.com/datasets/littleotter/united-states-presidential-speeches?select=presidential_speeches.csv), who scraped the data from the [Miller Center](https://millercenter.org/the-presidency/presidential-speeches). 
### 5.3.1 Data
The data used in this assignment are speeches given by American Presidents. The dataset from Lilleberg contains eight CSV files but I am only using the _presidential_speeches.csv_. This CSV file contains all official presidential speeches until September 9th, 2019. The CSV file has seven columns (_Data, President, Party, Speech Title, Summary, Transcript, URL_). There are 867 speeches with the Party affiliation of either Democratic or Republican. There are in total 478 Democratic speeches and 389 Republican speeches. 
## 5.4 Packages
The scripts in this repository use the following packages:
-	**Pandas (version 1.5.3)** is used to read the data, create CSV files, and manipulate data frames.
-	**Matplotlib (version 3.7.1)** is used to create the elbow method plot and style the scatterplots.
-	**Seaborn (version 0.12.2)** is used to create scatterplots in collaboration with Matplotlib.
-	**NumPy (version 1.24.3)** is used to manipulate arrays.
-	**Scikit-Learn (version 1.2.2)** is used to import the following: _TFIDFVectorizer_ is used to find the most significant terms used in the documents. _KMeans_ is an algorithm for clustering data points. _Metrics_ is used to evaluate the clustering algorithms performance. _PCA_ is used to create smaller dimensions of the data, making it easier to interpret the plots created.
-	**Re** is used to create regular expressions to clean the data.
-	**Os** is used to navigate across operating systems.
-	**Sys** is used to navigate the directory.
-	**Zipfile** is used to unpack the zip file.
-	**Argparse** is used to create command-line arguments.
## 5.5 Repository Contents
This repository contains the following folders and files:
-	***Data*** is an empty folder where the zip file will be placed.
-	***Figs*** is the folder that contains the cluster plots, and the visualisation of the _elbow method_.
-	***Out*** is the folder that contains the tables created in both scripts. The tables are the top keywords for the _elbow method_ clusters, for the _DBI/Silhouette score_ clusters, and for the _DBI/Silhouette score_ for clusters ranging in size from two to ten.
-	***Src*** is the folder that contains the two scripts. The script ``dbi_sil_elbow.py`` creates the _elbow method_ visualisation and the _DBI/Silhouette score_ table. The script ``kmeans_clustering.py`` creates the two cluster plots, and the top keywords tables. 
-	***README.md*** is the README file.
-	***Requirements.txt*** is a text file with version-controlled packages that are needed to run the scripts.
-	***Setup.sh*** is the file that creates a virtual environment, upgrades pip, and installs the packages from _requirements.txt_.
## 5.6 Methods 
### 5.6.1 Explanation of key methods
-	***Term frequency and inverse document frequency (TF-IDF):*** _TF-IDF_ is used to find the most meaningful words in a document. _TF_ is done by calculating the number of occurrences of a word in a document. _IDF_ is done by dividing the total number of documents by the number of documents the word appears in. This gives common words a low score and important words a high score. _TF-IDF_ is calculated by multiplying the _TF_ with the _IDF_. By using _TF-IDF_, meaningful words are given a high score and frequent words, like stop words, are given a low score.
-	***Davies-Bouldin Index (DBI):*** _DBI_ is used to evaluate how well the clustering algorithm performed. _DBI_ looks at how similar and compact a point in the cluster is, and how far away other clusters are. The _DBI_ score ranges from zero to infinity where a score closer to zero is best.
-	***Silhouette score:*** The _Silhouette score_ is also used to evaluate how well the clustering algorithm performed. The _Silhouette score_ is calculated by looking at how compact each cluster is and how far away one cluster is from the nearest cluster. A good clustering algorithm has a score of 1 and a bad clustering algorithm has a score of -1 -[source](https://towardsdatascience.com/three-performance-evaluation-metrics-of-clustering-when-ground-truth-labels-are-not-available-ee08cb3ff4fb).
-	***Elbow Method and Within-Cluster Sum of Squares (WCSS):***  The _elbow method_ is used to find the optimal number of clusters to define when using _KMeans_. The _elbow method_ is done by plotting clusters and their _WCSS_. The _WCSS_ is the compactness of the specified clusters. By plotting the _WCSS_ for different numbers of clusters (2-10), the optimal number of clusters can be found by looking at the point where the line plot takes a bend. This illustrates that more clusters will not significantly decrease the _WCSS_ - [source](https://www.analyticsvidhya.com/blog/2021/01/in-depth-intuition-of-k-means-clustering-algorithm-in-machine-learning/).
-	***KMeans:*** _KMeans_ is a popular algorithm for creating clustering. _KMeans_ works by randomly giving a number of cluster centroids. The algorithm then loops through the data looking for similarities in the data and placing them in one of the clusters. The cluster centroids are updated throughout this process, to be in the middle of the cluster – [source](https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1). 
-	***Principal Component Analysis (PCA):*** PCA is used to reduce the dimension size of the data. This is done to allow for easier interpretation of the visualisations created. However, using PCA does decrease the accuracy since values get reduced in size – [source](https://builtin.com/data-science/step-step-explanation-principal-component-analysis). 
### 5.6.2 Script dbi_elbow.py
-	This script starts by initializing two arg-parses for the path to the zip file and the number of features to use when creating the _TF-IDF vectorizer_. 
-	The CSV files in the zip file are then extracted. 
-	The CSV file is loaded in as a Pandas data frame and the data frame is reduced to only include rows that have _Democratic_ or _Republican_ in the column _Party_. 
-	A function is then created to clean the speeches. The clean function removes all words in-between parentheses and the parentheses themselves. This is done because it is not part of the speech, but a part of the transcript to show when people applauded the president. 
-	The clean function is then applied to each speech and the cleaned data frame is stored as a CSV file in the data folder to be used in the other script.
-	The _TF-IDF vectorizer_ is then initialized to remove stopwords, only keep the top number of features (the argparse), and unigrams and bigrams.
-	The cleaned speeches are then fitted and transformed to the vectorizer. 
-	After vectorizing the speeches, the function ``dbi_function`` is created. The function creates a table of _DBI_ and _Silhouette scores_ for clusters ranging in size from two to ten clusters. This is done by making a loop that loops over the _KMeans_ _algorithm_ with different cluster sizes and appending the _DBI_ and _Silhouette_ _score_ to an empty list. The table is then stored as a CSV file in the folder _out_. This function can be run with different max feature sizes to see how the scores change. I ran it with a feature size of 10.
-	The next function, ``elbow_function``, does the same as the ``dbi_function``, but calculates the _WCSS_ instead of the _DBI_ and _Silhouette_ _score_. 
-	Using the _WCSS_ the _elbow method_ plot is created, and stored in the folder _figs_. This function can also be run with different max feature sizes to see where the “elbow” occurs. I ran it with ten features and the “elbow” appeared at four clusters.
### 5.6.3 Script kmeans_clustering.py
-	This script starts by initializing three arg-parses. One for how many max features. One for how many clusters based on the _elbow method_. And one for how many clusters based on the _DBI_ and _Silhouette score_. 
-	The cleaned data frame from the previous script is then loaded using Pandas. 
-	The TF-IDF vectorizer is then initialized with the same arguments and the cleaned speeches are fitted to the vectorizer. 
-	The _KMeans_ _algorithm_ is then applied to the vectorized cleaned data with the number of clusters specified with argparse. To find the optimal number of clusters I looked at the _DBI_ and _Silhouette score table_ created in the previous script and decided to go with two clusters. 
-	The dimensions of the data are then reduced using _PCA_.
-	The cluster labels and the new dimensions are added to the data frame. 
-	Then the function ``get_top_keywords`` is created. This function finds the top two most meaningful terms of each cluster. This function is inspired by [D’Agostino](https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7).
-	The two most meaningful terms for each cluster are then stored and saved as a table in the folder _out_. 
-	Then the scatterplot of the clusters is created, by plotting the dimensions created by _PCA_ and with the clusters as the _hue_. Furthermore, the points have a unique marker to show if they are _Democratic_ or _Republican_.
-	The six previous steps are then repeated, but this time with the number of clusters deduced from the _elbow method_.
## 5.7 Discussion
### 5.7.1 DBI, Silhouette score, and Elbow Method.
Finding the optimal number of clusters is key to ensuring an accurate representation of the data. The _elbow method and DBI / Silhouette_ score give a good basis for choosing the optimal number of clusters. However, the two methods gave me two different numbers of clusters.
|                         | **10 Features** | **50 Features** | **100 Features** |
|-------------------------|-------------|-------------|--------------|
| **DBI / Silhouette Score**  | 2 clusters  | 2 clusters  | 2 clusters   |
| **Elbow Method**            | 4 clusters  | 5 clusters  | 5 clusters   |

The table shows the optimal number of clusters from the two different methods based on different max feature sizes. It also shows how the _DBI and Silhouette scores_ do not give the same number of clusters as the elbow method. 
### 5.7.2 Feature size
Increasing the number of features impacts the number of clusters. By having more features it can harm or benefit the number of clusters. It can harm by introducing irrelevant noise. The noise is words that do not carry special meaning but instead blur the clusters together. But, it can benefit the clusters by giving more words to cluster the speeches, providing that the words are unique to each cluster.  
### 5.7.3 Influencing factors 
The speeches are not solely transcripts of the United States President talking. The speeches include questions asked by the press and remarks given by Vice Presidents and Secretaries. This does not influence the results in a bad way, but does influence the results as the transcribes of other people may include words that blur the speeches together. 
Furthermore, the table of the top two terms for each cluster does not include bigrams but instead takes only takes single words. This causes table to have one cluster with the top two terms “United” and “States” instead of the bigram “United States.”
### 5.7.4 Cluster Scatterplots
The scatterplot cluster for the _DBI and Silhouette score_ shows two separate clusters that do not overlap but that are not compact. Cluster 0 has the keywords “United” and “States.” Cluster 1 has the keywords “World” and “People. 
The scatterplot for the _elbow method_ shows four clusters with minimal overlapping. The clusters that overlap the most are Cluster 0 and Cluster 2. The keywords for the clusters are.
|         | **Keyword One** | **Keyword Two** |
|---------|-------------|-------------|
| **Cluster 0** | American    | People      |
| **Cluster 1** | United      | States      |
| **Cluster 2** | People      | World       |
| **Cluster 3** | Congress    | Government  |


Both scatterplots of clusters do not have a big difference in _Democratic_ and _Republican_ clusters. The reason for that could be that the United States Presidents have some of the same speech topics (_Inaugural Address, State of the Union Address, and so on_). 
There might be a clearer cluster difference by grouping the speeches by title for the Presidents and looking at how speeches with the same topic may cluster.

## 5.8 Usage
To use the scripts in this repository, follow these steps:

**OBS! It is important to start with script ``dbi_sil_elbow.py`` as this is the script that unzips that zip file and produces the cleaned data frame for the other script.**

**OBS! The defaults that are in place for the argparse are the values I used to create the plots and tables in the folders _figs_ and _out_.**

### 5.8.1 Script dbi_sil_elbow.py
1.	Clone the repository.
2.	Navigate to the correct directory.
3.	Get the data from [Kaggle](https://www.kaggle.com/datasets/littleotter/united-states-presidential-speeches?select=presidential_speeches.csv) as a zip file and place it in the data folder (you might need to rename the zip file).
4.	Run ``bash setup.sh`` in the command line. This will create a virtual environment and install the requirements.
5.	Run ``source ./assignment_5/bin/activate`` in the command-line, to activate the virtual environment.
6.	In the command line write this ``python3 src/dbi_sil_elbow.py --zip_path data/archive.zip --features 10``
    - The argparse ``--zip_path`` takes a string as input. Here you must write the path to your zip file which should be located in the folder data.
    - The argparse ``--features`` takes an integer as input and has a default of 10. You can change this to see how a different number of max features changes the _DBI / Silhouette score_ and the _elbow method_ visualisation. 
### 5.8.2 Script kmeans_clustering.py
1.	In the command line write this ``python3 src/kmeans_clustering.py --features 10 --clusters_elbow 4 --clusters_dbi 2``
    - The argparse ``--features`` takes an integer as input and has a default of 10. **You should change this to match the number of features you chose in the other script**. 
    - The argparse ``--clusters_elbow`` takes an integer as input and has a default of 4. You should change this to match the number of clusters you deduced from the _elbow method_ visualisation. However, the value can not be below two.
    - The argparse ``--clusters_dbi`` takes an integer as input and has a default of 2. You should change this to match the number of clusters you deduced from the _DBI and Silhouette scores_. However, the value can not be below two.
