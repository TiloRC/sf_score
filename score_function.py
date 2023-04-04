def d(x1,x2):
    # euclidean distance squared
    return np.dot(x1-x2,x1-x2)

def score_function(X, kMeans,weight=1):
    """
    Calculates a score value for the quality of the clustering performed by k-means algorithm.
    Adjust the weight if the score value is the same for all clusterings. 
    Comes from this paper: https://www.semanticscholar.org/paper/A-Bounded-Index-for-Cluster-Validity-Saitta-Raphael/2fb5b1707e5ebec72ff8a0c8e75dd0fd00256d8d?p2df
    
    Parameters:
        X (numpy array): A 2D array of shape (n_samples, n_features) containing the data points that were clustered.
        kMeans (sklearn.cluster.KMeans): A KMeans object representing the k-means clustering performed on the data points.
        weight (float, optional): Adjust to avoid underflow or overflow.
    
    Returns:
        score (float): A score value that ranges from 0 to 1, where a higher score indicates better clustering performance.
    """
    
    
    k = kMeans.n_clusters
    clusters = group_by_cluster(X, kMeans.labels_, k)
    labs =  kMeans.labels_
    centroids = kMeans.cluster_centers_
    
    z_tot = sum(X)/len(X)
    
    def bcd():
        bcd = 0
        for i in range(k):
            clust_i = X[labs == i]
            bcd += d(centroids[i],z_tot)*len(clust_i)
        return bcd/(len(X)*k)
    
    def wcd():
        wcd = 0
        for i in range(k):
            clust_i = X[labs == i]
            wcd += math.sqrt(sum([d(x, centroids[i]) for x in clust_i])/len(clust_i))
        return wcd/k

    bcd = bcd(); wcd = wcd();  
    return 1 - 1/(np.e**(np.e**((bcd-wcd)/weight)  )) 
