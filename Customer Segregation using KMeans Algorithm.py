import numpy as np # Importing the NumPy library for numerical computing
import pandas as pd # Importing the Pandas library for data manipulation and analysis
import matplotlib.pyplot as plt # Importing the Matplotlib library for creating plots and visualizations
import seaborn as sns # Importing the Seaborn library for statistical data visualization
from sklearn.cluster import KMeans # Importing the KMeans class from scikit-learn for clustering
customer_data=pd.read_csv('/content/Mall_Customers.csv') #Customer database that will be used for training and testing
customer_data.head() # customer_data.head() is to quickly inspect the initial rows of the DataFrame and get a sense of the data structure and the values present in the columns. 
customer_data.shape # It allows you to quickly check the number of rows and columns in the DataFrame, which is essential for understanding the data and performing data manipulations and analyses.
customer_data.info() #Gives you information regarding the dataset
customer_data.isnull().sum() # The result of customer_data.isnull().sum() is a Series that shows the count of missing values in each column. Each column name is paired with the count of missing values in that column.
X=customer_data.iloc[:,[3,4]].values ##Choosing the Annual Income Column and Spending Score Column
print(X)
##WCSS-Within Clusters Sum of Squares is Calculated
# Finding wcss value for different number of clusters
wcss=[]
for i in range(1,11):
  kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
  kmeans.fit(X)
  wcss.append(kmeans.inertia_)
## Code for Elbow Point Graph to identify the minmum number of clusters that is required for the Kmeans clustering algorithm
sns.set()# Set the default style of Seaborn plots
plt.plot(range(1,11),wcss)# Create a line plot of the within-cluster sum of squares (WCSS) against the number of clusters
plt.title("The Elbow Point Graph")# Set the title of the plot
plt.xlabel("Number of Clusters")# Set the label for the x-axis
plt.ylabel("WCSS")# Set the label for the y-axis
plt.show()# Display the plot
"""
The elbow point graph is plotted to determine the optimal number of clusters in a K-means clustering algorithm. Here's why it is useful:

In K-means clustering, the goal is to partition a dataset into a specified number of clusters (K) based on the similarity of data points. However, determining the appropriate value of K is not always straightforward. The elbow point graph helps in finding an optimal value for K.

The graph is created by plotting the within-cluster sum of squares (WCSS) against the number of clusters. WCSS is a measure of the variability or dispersion of data points within each cluster. It quantifies how close the data points are to their respective cluster centroids.

The idea behind the elbow point graph is to identify a point where adding more clusters does not significantly improve the clustering quality. The plot usually forms a downward curve, resembling an elbow. The "elbow point" on the graph represents the value of K where the rate of decrease in WCSS starts to level off. This indicates a diminishing return in performance improvement by adding more clusters.

The elbow point is considered as a reasonable choice for the number of clusters since it balances the trade-off between model complexity (more clusters) and the goodness of fit (lower WCSS). It helps avoid overfitting or underfitting the data by selecting an appropriate level of clustering.

By analyzing the elbow point graph, one can make an informed decision on the optimal number of clusters to use in the K-means clustering algorithm.
"""""
### Code for KMEANS Clustering Algorithm
kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0) ## 5 clusters are formed for kmeans algorithm

Y=kmeans.fit_predict(X)

print(Y)
##Plotting the graph of results of KMeans clustering Algorithm that segregates the customers into different groupsW
#Plotting all the clusters and their Centroids

plt.figure(figsize=(8,8))
plt.scatter(X[Y==0,0],X[Y==0,1],s=50,c='green',label='Cluster 1')
plt.scatter(X[Y==1,0],X[Y==1,1],s=50,c='red',label='Cluster 2')
plt.scatter(X[Y==2,0],X[Y==2,1],s=50,c='yellow',label='Cluster 3')
plt.scatter(X[Y==3,0],X[Y==3,1],s=50,c='violet',label='Cluster 4')
plt.scatter(X[Y==4,0],X[Y==4,1],s=50,c='blue',label='Cluster 5')
#Plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='cyan',label='Centroids')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')
plt.show()