# Importing data manipulation library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import string 

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
from sklearn.decomposition import PCA


def input_parse():
    # Command line arguments 
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip_path", type=str, help = "Name of the zip folder")
    args = parser.parse_args()

    return args



def unzip(args):
    filepath = os.path.join("..","data","presidential_speeches.csv")
    if not os.path.exists(filepath): # If the folder path does not exist, unzip the folder, if it exists do nothing 
        print("Unzipping file")
        path_to_zip = args.zip_path # Defining the path to the zip file
        zip_destination = os.path.join("..","data") # defining the output destination

        with zipfile.ZipFile(path_to_zip,"r") as zip_ref: # using the package from zipfile, to un zip the zip file
            zip_ref.extractall(zip_destination) # Unzipping
    print("The files are unzipped")
    return filepath


def load_dataframe(filepath):
    corpus = pd.read_csv(filepath)
    corpus_true = corpus[corpus['Party'].isin(['Democratic', 'Republican'])]
    return corpus_true


def clean_function(corpus): 
    text = re.sub(r'\([^)]*\)', '', corpus) # Remove text between parentheses
    text = re.sub("[^A-Za-z]+", " ", text)  # Removes everything that is not words.
    # Return text in lowercase and stripped of whitespaces
    text = text.lower().strip()
    return text


def clean_data(corpus_true):
    corpus_true['cleaned'] = corpus_true['Transcript'].apply(lambda x: clean_function(x))
    return corpus_true


def assign_speeches(corpus_true):
    X = corpus_true["cleaned"]
    return X


def vectorizer_function():
    vectorizer = TfidfVectorizer(
                            stop_words = "english",
                            max_features = 10,
                            ngram_range = (1,2),
                            min_df = 5,
                            max_df = 0.95)
    return vectorizer


def fit_data(vectorizer, X):
    X_feats = vectorizer.fit_transform(X)
    return X_feats


def kmeans_function(X_feats, number):
    # Create Kmeans object and fit it to the training data
    number_of_clusters = number  
    kmeans = KMeans(n_clusters=number_of_clusters).fit(X_feats)
    # Get the labels using KMeans
    pred_labels = kmeans.labels_
    return kmeans, pred_labels, number_of_clusters


def dbi_function(X_feats, sil_pred_labels):
    dbi = metrics.davies_bouldin_score(X_feats.toarray(), sil_pred_labels)
    # Compute Silhoutte Score
    ss = metrics.silhouette_score(X_feats.toarray(), sil_pred_labels , metric='euclidean')
    # Print the DBI and Silhoutte Scores
    print("DBI Score: ", dbi, "\nSilhoutte Score: ", ss)
 
def elbow_function(X_feats):
    max_clusters = 10
    wcss = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans.fit(X_feats)
        wcss.append(kmeans.inertia_)
    return max_clusters, wcss

def elbow_plot(max_clusters, wcss):
    # Plot the WCSS values
    plt.plot(range(2, max_clusters + 1), wcss)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
    plt.title('WCSS for Different Numbers of Clusters')
    filename = os.path.join("figs", "Elbow" + '.png') # Saving the plot
    plt.savefig(filename)
    plt.clf() # Clearing plot
   

def pca_function(X_feats, number_of_clusters):
    # initialize PCA with 2 components
    pca = PCA(n_components=number_of_clusters, random_state=42)
    # pass our X to the pca and store the reduced vectors into pca_vecs
    pca_vecs = pca.fit_transform(X_feats.toarray())
    # save our two dimensions into x0 and x1
    x0 = pca_vecs[:, 0]
    x1 = pca_vecs[:, 1]
    return x0, x1

def pca_dataframe(corpus_true, x0, x1, pred_labels):
    corpus_true['cluster'] = pred_labels
    corpus_true['x0'] = x0
    corpus_true['x1'] = x1
    return corpus_true


def mapping(corpus_true, sil = bool):
    # map clusters to appropriate labels 
    if sil:
        cluster_map = {0: "1", 1: "2"}
    else: 
        cluster_map = {0: "1", 1: "2", 2: "3", 3: "4"}
    # apply mapping
    corpus_true['cluster'] = corpus_true['cluster'].map(cluster_map)   
    return corpus_true


def cluster_plot(corpus_true, name):
    # set image size
    plt.figure(figsize=(12, 7))
    # set a title
    plt.title("Speeches Tfidf", fontdict={"fontsize": 18})
    # set axes names
    plt.xlabel("X0", fontdict={"fontsize": 16})
    plt.ylabel("X1", fontdict={"fontsize": 16})
    # create scatter plot with seaborn, where hue is the class used to group the data
    sns.scatterplot(data=corpus_true, x='x0', y='x1', hue='cluster', palette="viridis", style='Party', markers={'Democratic': 'd', 'Republican': 's'})
    filename = os.path.join("figs", name + '.png') # Saving the plot
    plt.savefig(filename)
    plt.clf() # Clearing plot

def main_function():
    args = input_parse()
    filepath = unzip(args)
    corpus_true = load_dataframe(filepath)
    corpus_true = clean_data(corpus_true)
    X = assign_speeches(corpus_true)
    vectorizer = vectorizer_function() 
    X_feats = fit_data(vectorizer, X)
    sil_kmeans, sil_pred_labels, number_of_clusters = kmeans_function(X_feats, 2)
    dbi_function(X_feats, sil_pred_labels)
    x0, x1 = pca_function(X_feats, number_of_clusters)
    corpus_true = pca_dataframe(corpus_true, x0, x1, sil_pred_labels)
    corpus_true = mapping(corpus_true, sil = True)
    cluster_plot(corpus_true, "no_maxf_sil_cluster")
    elbow_kmeans, elbow_pred_labels, number_of_clusters = kmeans_function(X_feats, 4)
    max_clusters, wcss = elbow_function(X_feats)
    elbow_plot(max_clusters, wcss)
    x0, x1 = pca_function(X_feats, number_of_clusters)
    corpus_true = pca_dataframe(corpus_true, x0, x1, elbow_pred_labels)
    corpus_true = mapping(corpus_true, sil = False)
    cluster_plot(corpus_true, "no_maxf_elbow_cluster")

if __name__ == "__main__": # If called from commandline run main function
    main_function()