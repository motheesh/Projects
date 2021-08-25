from sklearn.cluster import KMeans
from kneed import KneeLocator
import pandas as pd
import pickle
from ApplicationLogger.logger import logger

class clusterData:
    def __init__(self):
        self.logPath="./TrainingLog/TrainingModelLog"
        self.logger=logger(self.logPath)
        try:
            self.logger.log("info",f"DATA CLUSTERING: starting initialization of data Clustering")
            self.clusterModelPath="./DataClustering/model/clusterData.pkl"
            self.logger.log("info",f"DATA CLUSTERING: Ending initialization of data Clustering")
        except Exception as e:
            self.logger.log("error",f"DATA CLUSTERING: Error while initializing data Clustering ERROR:{e}")
            raise Exception(e)
        
    def getK_forCluster(self,X):
        try:
            self.logger.log("info",f"DATA CLUSTERING: started getting K value for cluster")
            inertia_list=[]
            for i in range(1,10):
                kmeans=KMeans(n_clusters=i,init="k-means++", random_state=23)
                kmeans.fit(X)
                inertia_list.append(kmeans.inertia_)
            #plt.plot(range(1,10),inertia_list)
            #plt.xlabel("clusters")
            #plt.ylabel("Intertia")
            #plt.show()
            kn=KneeLocator(range(1,10),inertia_list,curve="convex",direction="decreasing")
            n_clstr=kn.knee
            self.logger.log("info",f"DATA CLUSTERING: Ended getting K value for cluster")
            return n_clstr
        except Exception as e:
            self.logger.log("error",f"DATA CLUSTERING: Error while getting K value for cluste ERROR:{e}")
            raise Exception(e)
   
    
    def PredictCluster(self,x):
        try:
            self.logger.log("info",f"DATA CLUSTERING: started predicting data cluster using saved model")
            with open(self.clusterModelPath, 'rb') as file:
                # Call load method to deserialze
                ClusterModel = pickle.load(file)
                clusters=ClusterModel.fit_predict(x)
                x["clusters"]=clusters
                self.logger.log("info",f"DATA CLUSTERING: Ended predicting data cluster using saved model")
                return x
        except Exception as e:
            self.logger.log("error",f"DATA CLUSTERING: Error while predicting data cluster using saved model ERROR:{e}")
            raise Exception(e)
             
    def ClusterData(self,X,Y):
        try:
            self.logger.log("info",f"DATA CLUSTERING: started clustering data for training")
            n_clstr=self.getK_forCluster(X)
            kmeans=KMeans(n_clusters=n_clstr,init="k-means++", random_state=23)
            details=kmeans.fit_predict(X)
            self.saveClusterModel(kmeans)
            X["clusters"]=pd.Series(details)
            X["Target"]=Y
            self.logger.log("info",f"DATA CLUSTERING: Ended clustering data for training")
            return X,n_clstr
        except Exception as e:
            self.logger.log("error",f"DATA CLUSTERING: Error while clustering data for training ERROR:{e}")
            raise Exception(e)
   
    
    def saveClusterModel(self,model):
        try:
            self.logger.log("info",f"DATA CLUSTERING: started saving clustering model for prediction")
            pickle.dump(model,open(self.clusterModelPath,'wb'))
            self.logger.log("info",f"DATA CLUSTERING: Ended saving clustering model for prediction")
            return True
        except Exception as e:
            self.logger.log("error",f"DATA CLUSTERING: Error while saving clustering model for prediction ERROR:{e}")
            raise Exception(e)
   