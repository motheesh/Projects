import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import Lasso,LassoCV
from sklearn.impute import KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import VarianceThreshold
import pickle
from ApplicationLogger.logger import logger
#Author: Motheeshkumar Jay 
#data Preprocessing

class dataPreprocessing:
    def __init__(self):
        self.logPath="./TrainingLog/TrainingModelLog"
        self.logger=logger(self.logPath)
        try:
            self.logger.log("info",f"DATA PREPROCESSING: starting initialization of data preprocessing")
            self.trainPath="./TrainingInputData/insurance.csv"
            self.predictPath="./PredictionInputData/insurance.csv"
            self.ImputeModelPath="./DataPreparation/ImputeDataModel"
            self.AfterChangeDataStructPath="./DatasetSchema/AfterChanges"
            self.logger.log("info",f"DATA PREPROCESSING: Ending initialization data preprocessing")
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while initializing data preprocessing ERROR:{e}")
            raise Exception(e)
        
    def removeConstantFeatures(self,x):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: started removing constant features if any")
            columns=x.columns
            selector=VarianceThreshold(threshold=0.01)
            selector.fit(x)
            valid_features=columns[selector.get_support()]
            self.logger.log("info",f"DATA PREPROCESSING: Ended removing constant features if any")
            return x[valid_features]
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while removing constant Features ERROR:{e}")
            raise Exception(e)
        
    def get_outliers(self,df,feature):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: Started getting outliers index")
            q1=df[feature].quantile(.25)
            q3=df[feature].quantile(.75)
            IQR=q3-q1
            lowerBound=q1-(1.5*IQR)
            upperBound=q3+(1.5*IQR)
            self.logger.log("info",f"DATA PREPROCESSING: Ended getting outliers index")
            return df.index[(df[feature]<lowerBound)|(df[feature]>upperBound)]
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while removing outliers ERROR:{e}")
            raise Exception(e)


    def remove_outliers(self,df,target):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: Started removing outliers")
            outlier_index=[]
            for i in df.dtypes[df.dtypes!=object].keys():
                if target!=i:
                    index=self.get_outliers(df,i)
                    if len(index)>0:
                        outlier_index.extend(index)
            outlier_index=set(outlier_index)
            if len(outlier_index)>0:
                df=df.drop(outlier_index)
            self.logger.log("info",f"DATA PREPROCESSING: Ended removing outliers")
            return df
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while removing outliers ERROR:{e}")
            raise Exception(e)

    def getTrainingData(self):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: Started getting Training data")
            data=pd.read_csv(self.trainPath)
            Target=data.iloc[:,-1].name
            #data=self.remove_outliers(data,Target)
            X=data.iloc[:,:-1]
            Y=data.iloc[:,-1]
            self.logger.log("info",f"DATA PREPROCESSING: Ended getting Training data")
            return X,Y

        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while getting Training data ERROR:{e}")
            raise Exception(e)

    def GetTemplateDetails(self):
        X,_=self.getTrainingData()
        features=X.columns
        dataTypes=X.dtypes
        dict_={}
        count=0
        for i,j in zip(features,dataTypes):
            if j==object:
                print(f"{i} select")
                dict_[count]={"label":i,"tag":"select","data":X[i].unique()}
            else:
                dict_[count]={"label":i,"tag":"input","data":""}
            count=count+1
        return dict_


    def getPredictionData(self):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: Started getting Prediction data")
            X=pd.read_csv(self.predictPath)
            self.logger.log("info",f"DATA PREPROCESSING: Ended getting Prediction data")
            return X
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while getting Prediction data ERROR:{e}")
            raise Exception(e)


    def preserveColumnNamesForPrediction(self,df):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: Started preserving column structure after all changes")
            features=df.columns
            path=f"{self.AfterChangeDataStructPath}/model.pkl"
            pickle.dump(features,open(path,"wb"))
            self.logger.log("info",f"DATA PREPROCESSING: Ended preserving column structure after all changes")
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while preserving column structure after all changes ERROR:{e}")
            raise Exception(e)

    def encodeCategoricalData(self,X,drop_first=1):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: Started Encoding Categorical data")
            categorical=np.array(X.dtypes[X.dtypes==object].index)
            X=pd.get_dummies(X,drop_first=drop_first)
            self.logger.log("info",f"DATA PREPROCESSING: Ended Encoding Categorical data")
            return X
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while Encoding Categorical data ERROR:{e}")
            raise Exception(e)

    
    
    def saveImputeDataModel(self,model):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: Started Saving data Imputation model")
            path=f"{self.ImputeModelPath}/model.pkl"
            pickle.dump(model,open(path,"wb"))
            self.logger.log("info",f"DATA PREPROCESSING: Ended Saving data Imputation model")
            return True
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while Saving data Imputation model ERROR:{e}")
            raise Exception(e)
    
        
    
    def imputeNullValues(self,X):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: Started Imputing NULL Values")
            totalNullValues=sum(X.isna().sum().values)
            imputer=KNNImputer(n_neighbors=5,weights="distance")
            imputer.fit(X)
            if totalNullValues!=0:
                X=imputer.transform(X)
            self.saveImputeDataModel(imputer)
            self.logger.log("info",f"DATA PREPROCESSING: Started Imputing NULL Values")
            return X
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while Imputing NULL Values ERROR:{e}")
            raise Exception(e)
    

    def featureSelection(self,x,y):
        try:
            self.logger.log("info",f"DATA PREPROCESSING: Started feature selection using Lasso coeff")
            features=x.columns
            lasso=Lasso(alpha=100)
            lasso.fit(x,y)
            selected_features=features[np.where(lasso.coef_!=0)]
            self.logger.log("info",f"DATA PREPROCESSING: Ended feature selection using Lasso coeff")
            return selected_features
        except Exception as e:
            self.logger.log("error",f"DATA PREPROCESSING: Error while selecting features using Lasso coeff ERROR:{e}")
            raise Exception(e)
    

