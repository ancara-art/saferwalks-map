from sklearn.cluster import KMeans
from sklearn import metrics
import pandas as pd
import hdbscan


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


def evaluation_metrics(X, labels_pred, metric_name):
    """
    Function for calculating the accuracy of the Kmeans algorithms
    """

    s_s = metrics.silhouette_score(X, labels_pred, metric=metric_name)
    
    metrics_names = ["Silh_S"]
    values = [s_s]
    
    result = list(zip(metrics_names,values))
    result = pd.DataFrame(result, columns=['Metric','Value'])

    return result
