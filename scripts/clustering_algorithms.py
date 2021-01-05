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
    Function for clustering with algorithm hdbscan. Currently we are not using this one.
    """

    hdbscan_cluster = hdbscan.HDBSCAN(min_cluster_size=n_clusters, metric = 'haversine')
    hdbscan_cluster.fit(X)

    return hdbscan_cluster.labels_ 

def agglomerative(X, n_clusters):
    """
    Function for clustering with algorithm Agglomerative clustering.
    """
    agglo_cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='ward')#Ward and average gives same result. Single linkage is worst.
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
    
    metricnames = ['Silhouette Score']
    values = [s_s]
    dataframe_with_metrics = pd.DataFrame({'Metric': metricnames, 'Value': values, 'Algorithm':[algorithm]})

    return dataframe_with_metrics

def validate_number_of_points(df, n_clusters):
    """
    Function for validating that the number of points (parents) for a school, is greater than 
    the number of clusters given by the user in the input box. 
    """
    validation = len(df[['user_long', 'user_lat']].sum(axis=1).unique()) > n_clusters

    return validation