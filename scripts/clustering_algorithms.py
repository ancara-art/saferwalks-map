from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import hdbscan
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import SpectralClustering


def filter_school(df, school_Id):
    """
    We are filtering the merged data frame (parents and schools) by the school selected 
    in the map and returning the dff with the three columns (user_long, user_lat, and 
    parentName attributes).
    """
    dff = df.loc[df['schoolId'] == school_Id, ['user_long', 'user_lat', 'parentName', 
                                               'schoolId', 'school_long', 'school_lat', 'school_name']] 

    return dff

def kmeans(X, n_clusters):
    """
    Function for clustering with algorithm kmeans.
    """

    kmeans_cluster = KMeans(n_clusters=n_clusters)
    #compute kmeans clustering
    kmeans_cluster.fit(X)
    
    return kmeans_cluster.labels_

def hdbscan_algorithm(X, n_clusters):
    """
    Function for clustering with algorithm hdbscan.
    """

    hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=n_clusters, metric = 'haversine')
    hdbscan_cluster.fit(X)

    return hdbscan_cluster.labels_ 


def agglomerative(X, n_clusters):
    """
    Function for clustering with algorithm Agglomerative clustering.
    """

    agglo_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')
    agglo_cluster.fit(X)

    return agglo_cluster.labels_ 
                                        
    
    
def spectral(X, n_clusters):
    """
    Function for clustering with algorithm Spectral clustering.
    """

    spectral_cluster = SpectralClustering(n_clusters=n_clusters)
    spectral_cluster.fit(X)

    return spectral_cluster.labels_ 
                                            

def evaluation_metrics(X, labels_pred, metric_name, algorithm):
    """
    Function for calculating the accuracy of the algorithms for clustering.
    """
    s_s = metrics.silhouette_score(X, labels_pred, metric=metric_name)
    
    metricnames = ['Silhoutte Score']
    values = [s_s]
    dataframe_with_metrics = pd.DataFrame({'Metric': metricnames, 'Value': values, 'Algorithm':[algorithm]})

    return dataframe_with_metrics