# Importing data manipulation library
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
    parser.add_argument("--features", type=int, default= 10, help = "How many max features in TF-IDF vectorizer") 
    parser.add_argument("--clusters_elbow", type=int, default= 4, help = "How many clusters using the Elbow Method")  
    parser.add_argument("--clusters_dbi", type=int, default = 2, help = "How many clusters according to the DBI and Silhouette score") 
    args = parser.parse_args()
    return args


def load_dataframe():
    print("Loading data")
    filepath = os.path.join("data", "cleaned_president_speeches.csv")
    corpus = pd.read_csv(filepath) # Loading the CSV into a data frame with Pandas
    X = corpus["cleaned"] # Placing the column clean in a new variable.
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


def dbi_kmeans_function(X_feats, args):
    print("Creating Kmeans from DBI Score")
    number = args.clusters_dbi
    number_of_clusters = number # Number of clusters. Specified in main function
    kmeans = KMeans(n_clusters=number_of_clusters).fit(X_feats) # Creating the Kmeans algorithm on the features with specified amount of clusters
    pred_labels = kmeans.labels_ # Getting the cluster labels for each speech 
    return kmeans, pred_labels, number_of_clusters


def elbow_kmeans_function(X_feats, args):
    print("Creating Kmeans from Elbow Method")
    number = args.clusters_elbow
    number_of_clusters = number # Number of clusters. Specified in main function
    kmeans = KMeans(n_clusters=number_of_clusters).fit(X_feats) # Creating the Kmeans algorithm on the features with specified amount of clusters
    pred_labels = kmeans.labels_ # Getting the cluster labels for each speech 
    return kmeans, pred_labels, number_of_clusters  


def pca_function(X_feats, number_of_clusters): # Reducing the dimensions of the data to create easier interpretation of the visualisations created later.
    print("Reducing dimensions with PCA")
    pca = PCA(n_components=number_of_clusters, random_state=42) # Initializing the PCA with specified number of clusters
    pca_vecs = pca.fit_transform(X_feats.toarray()) # Fitting and transforming the data to the reduced dimensionalities
    x0 = pca_vecs[:, 0] # Saving the first principle component 
    x1 = pca_vecs[:, 1] # Saving the second principle component
    return x0, x1


def pca_dataframe(corpus_cleaned, x0, x1, pred_labels):
    print("Adding columns Cluster, and PCA dimension, to the data frame")
    corpus_cleaned['cluster'] = pred_labels # Adding a column cluster to the dataframe that stores which cluster the speech belongs to
    corpus_cleaned['x0'] = x0 # Adding column with the first principle component 
    corpus_cleaned['x1'] = x1 # Adding column with the second principle componet 
    return corpus_cleaned


def get_top_keywords(n_terms, pred_labels, vectorizer, X_feats):
    """This function was inspired from https://medium.com/mlearning-ai/text-clustering-with-tf-idf-in-python-c94cd26a31e7"""
    df = pd.DataFrame(X_feats.todense()).groupby(pred_labels).mean() # creating data frame and grouping by cluster 
    terms = vectorizer.get_feature_names_out() # getting the features names from the TF-IDF vectorizer
    keyword = []  # Empty list to store keywords
    for i, r in df.iterrows(): # Loops through each row for each label
        cluster_keywords = [terms[t] for t in np.argsort(r)[-n_terms:]]  # Find the number of terms with the highest tf-idf score
        keyword.append({'Cluster': i, 'Keywords': cluster_keywords})  # Store cluster and keywords in a dictionary
    keyword_df = pd.DataFrame(keyword)  # Create DataFrame from the keyword list
    return keyword_df


def top_keywords_dataframe(pred_labels, vectorizer, X_feats, name, clusters):
    print("Getting top two keywords pr cluster")
    top_keywords_df = get_top_keywords(2, pred_labels, vectorizer, X_feats) # Using the above function on get the top two keywords for each cluster
    out_path = os.path.join("out", name + "_clusters_" + str(clusters)+ ".csv") # Save path and name
    top_keywords_df.to_csv(out_path, index=False) # Saving the data frame as csv


def cluster_plot(corpus_cluster, name, features, clusters):
    print("Creating plot")
    plt.figure(figsize=(12, 7)) # Setting the plot size
    plt.title("Speeches Tfidf", fontsize = "18") # Plot title
    plt.xlabel("Dimension 1", fontsize = "18") # X-axis name 
    plt.ylabel("Dimension 2", fontsize = "18") # Y-axis name
    sns.scatterplot(data=corpus_cluster, x='x0', y='x1', hue='cluster', palette="viridis", style='Party', markers={'Democratic': 'd', 'Republican': 's'}) # Creating the scatterplot. 
    # The style is which Party the speech belongs to. The dots have unique markers to show which party they belong to
    filename = os.path.join("figs", name + "_features_" + str(features) + "_clusters_" + str(clusters) +'.png') # Filename and file path
    plt.savefig(filename) # Saving the plot
    plt.clf() # Clearing plot


def main_function():
    args = input_parse() # Creating command line arguments
    corpus_cleaned, X = load_dataframe() # Loading the data as a Pandas data frame
    vectorizer, features = vectorizer_function(args) # Creating TF-IDF vectorizer
    X_feats = fit_data(vectorizer, X) # Fitting the data to the vectorizer

    sil_kmeans, sil_pred_labels, sil_number_of_clusters = dbi_kmeans_function(X_feats, args) # Using the Kmean algorithm to create clusters
    x0, x1 = pca_function(X_feats, sil_number_of_clusters) # Reducing dimensionality 
    corpus_cluster = pca_dataframe(corpus_cleaned, x0, x1, sil_pred_labels) # Adding PCA scores to dataframe
    top_keywords_dataframe(sil_pred_labels, vectorizer, X_feats, "dbi_top_keywords", sil_number_of_clusters) # Finding the two top feature names for the dbi clusters
    cluster_plot(corpus_cluster, "DBI__silhouette_cluster_plot", features, sil_number_of_clusters) # Creating the plot for the Silhouette score clusters

    elbow_kmeans, elbow_pred_labels, elbow_number_of_clusters = elbow_kmeans_function(X_feats, args) # Using the Kmean algorithm to create clusters
    x0, x1 = pca_function(X_feats, elbow_number_of_clusters) # Reducing dimensionality 
    corpus_cluster = pca_dataframe(corpus_cleaned, x0, x1, elbow_pred_labels) # Adding PCA scores to dataframe
    top_keywords_dataframe(elbow_pred_labels, vectorizer, X_feats, "elbow_top_keywords", elbow_number_of_clusters) # Finding the two top feature names for the elbow clusters
    cluster_plot(corpus_cluster, "elbow_method_cluster_plot", features, elbow_number_of_clusters) # Creating the plot for the elbow method score clusters


if __name__ == "__main__": # If called from commandline run main function
    main_function()