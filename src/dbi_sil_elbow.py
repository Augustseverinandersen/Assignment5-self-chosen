# Importing data manipulation library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns # Maybe remove
import numpy as np
import re
import string # maybe removeeeeeeeeeeeeeeeeeeeeeeeee

# Importing systems library
import os
import sys
sys.path.append(".")
import zipfile 
import argparse

# Importing from Sci-Kit Learn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA # Removeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee



def input_parse():
    # Command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, help = "Name of the zip folder") # Create argparse for path to zipfile 
    parser.add_argument("--features", type=int, default= 10, help = "How many max features in TF-IDF vectorizer") # Create argparse for path to zipfile
    args = parser.parse_args()
    return args


def unzip(args):
    filepath = os.path.join("data","presidential_speeches.csv")
    if not os.path.exists(filepath): # If the folder path does not exist, unzip the folder, if it exists do nothing 
        print("Unzipping file")
        path_to_zip = args.zip_path # Defining the path to the zip file
        zip_destination = os.path.join("data") # defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination) # Unzipping
    print("The files are unzipped")
    return filepath



def load_dataframe(filepath):
    print("Loading the data")
    corpus = pd.read_csv(filepath) # Loading the CSV into a data frame with Pandas
    corpus = corpus[corpus['Party'].isin(['Democratic', 'Republican'])] # Removing all rows that are not democratic or republican in column "Party"
    return corpus


def clean_function(corpus): 
    text = re.sub(r'\([^)]*\)', '', corpus) # Removing text between parentheses (unclear), (applause)
    text = re.sub("[^A-Za-z]+", " ", text)  # Removing everything that is not words.
    text = text.lower().strip() # Returning text in lowercase and removing whitespaces
    return text


def clean_data(corpus):
    print("Cleaning the data")
    corpus['cleaned'] = corpus['Transcript'].apply(lambda x: clean_function(x)) # Applying the clean function on each cell in the column "Transcript" and placing in new column "Cleaned"
    X = corpus["cleaned"] # Placing the column clean in a new variable.
    out_path = os.path.join("data", "cleaned_president_speeches.csv") # File path and file name
    corpus.to_csv(out_path, index=False) # Saving the cleaned csv file in the data folder.
    return corpus, X


def vectorizer_function(args):
    print("Creating vectorizer")
    features = args.features
    vectorizer = TfidfVectorizer( # Creating TF-IDF vectoriser
                            stop_words = "english", # Dont take English stopwords
                            max_features = features, # Only take the top specified number of features 
                            ngram_range = (1,2) # single words and bigrams.
                            )
    return vectorizer, features


def fit_data(vectorizer, X):
    print("Fitting the data to the vectorizer")
    X_feats = vectorizer.fit_transform(X) # Fitting and transforming the speeches to the vectorizer
    return X_feats


def dbi_function(X_feats, features):
    print("Creating table of DBI and silhouette scores")
    max_clusters = 10 # Max clusters
    dbi_score = [] # Empty list to store DBI score
    silhouette_score = [] # Empty list to store silhouette socre
    for n_clusters in range(2, max_clusters + 1): # For every cluster (starting at 2 to 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42) # Uses the Kmeans algorithm to create specified amount of clusters
        kmeans.fit(X_feats) # fits the algorithm to the vectorized data    
        labels = kmeans.labels_ # Storing the kmean labels
        dbi = metrics.davies_bouldin_score(X_feats.toarray(), labels) # Calculating the DBI score to see how good my clustering is
        ss = metrics.silhouette_score(X_feats.toarray(), labels , metric='euclidean') # Calculating the Silhouette score to see how good my cluster is.
        dbi_score.append(dbi) # Appending the dbi score to the empty list
        silhouette_score.append(ss) # Appending the silhouette score to the empty lsit
    DBI_Silhouette_score = pd.DataFrame({'DBI': dbi_score, 'Silhouette': silhouette_score}, index=range(2, max_clusters + 1)) # Creating a data frame of the DBI score and silhouette score. Index starting at 2 and ending at 10
    output_filepath = os.path.join("out", "features_" + str(features) +'_DBI_and_Silhouette_score.csv') # file path and file name
    DBI_Silhouette_score.to_csv(output_filepath, index_label='Clusters') # Saving the csv file to the folder out


def elbow_function(X_feats):
    print("Finding WCSS scores for elbow method")
    max_clusters = 10 # Defining max clusters
    wcss = [] # Creating empty list to store WCSS
    for n_clusters in range(2, max_clusters + 1): # for every cluster (starting at 2 to 10)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42) # Uses the Kmeans algorithm to create specified amount of clusters
        kmeans.fit(X_feats) # fits the algorithm to the vectorized data
        wcss.append(kmeans.inertia_) # Appends the WCSS score to the empty list.
    return max_clusters, wcss


def elbow_plot(max_clusters, wcss, features): 
    print("Plotting elbow method")
    plt.plot(range(2, max_clusters + 1), wcss, marker = "x") # Creating a plot to find the right amount of clusters with the Elbow Method
    plt.xlabel('Clusters', fontsize = "18") # X-axis label
    plt.ylabel('WCSS', fontsize = "18") # Y-axis label
    plt.title('WCSS for Different Numbers of Clusters', fontsize = "18") # Plot title
    filename = os.path.join("figs", "features_" + str(features) + "_Elbow_method" + '.png') # giving filename and path
    plt.savefig(filename) # Saving the plot
    plt.clf() # Clearing plot


def main_function():
    args = input_parse() # Command line arguments
    filepath = unzip(args) # Unzip function to unzip the zip folder
    corpus= load_dataframe(filepath) # Loading the CSV file to a dataframe, and removing rows.
    corpus, X = clean_data(corpus) # Cleaning the data
    vectorizer, features = vectorizer_function(args) # initializing the TF-IDF vectorizer 
    X_feats = fit_data(vectorizer, X) # Fitting and transforming the data to the vectorizer
    dbi_function(X_feats, features) # Creating a table of DBI and silhouette scores for cluster size 2-10
    max_clusters, wcss = elbow_function(X_feats) # Creating function to find WCSS for cluster size 2-10
    elbow_plot(max_clusters, wcss, features) # Plotting the elbow method.


if __name__ == "__main__": # If called from command line run main function
    main_function()
