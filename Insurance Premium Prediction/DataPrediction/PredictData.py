import pickle
import pandas as pd
from PredictionDataValidation.ValidatePredictionData import PredictValidation
from DataPreparation.dataPreprocessing import dataPreprocessing
from ApplicationLogger.logger import logger
import numpy as np
import os

class DataPrediction:
    def __init__(self):
        self.PredictionLogPath="./PredictionLog/DataPredictionLog"
        self.logger=logger(self.PredictionLogPath)
        try:
            self.logger.log("info","DATA PREDICTION: starting Initialization of DataPrediction object")     
            self.clusterPath="./DataClustering/model/clusterData.pkl"
            self.ModelPath="./TrainedModels"
            self.BatchTestResultPath="./PredictionBatchResult/InsurancePrediction.csv"
            self.ImputeModelPath=f"./DataPreparation/ImputeDataModel/model.pkl"
            self.goodDataPath="./PredictionGoodDataset"
            self.preserveStruct="./DatasetSchema/AfterChanges/model.pkl"
            self.prefix="DATA PREDICTION"
        except Exception as e:
            self.logger.log("error","DATA PREDICTION: error while Initializing DataPrediction object{e}")
            raise Exception(e)
        
    def validate(self):
        try:
            self.logger.log("info",f"{self.prefix}: started data validation")
            #initialize object for DataPrediction
            validate=PredictValidation()
            #validating prediction data
            validate.ValidatePredictData()
            
            self.logger.log("info",f"{self.prefix}: Ended data validation")
            return True
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while validating data ERROR:{e}")
            raise Exception(e)
    
    def getGoodData(self):
        try:
            self.logger.log("info",f"{self.prefix}: starting get good data")
            df=[]
            for fileName in os.listdir(self.goodDataPath):
                if fileName.__contains__("csv"):
                    temp=pd.read_csv(self.goodDataPath+"/"+fileName)
                    df.append(temp)
            dataset=pd.concat(df)
            self.logger.log("info",f"{self.prefix}: Ending get good data")
            return dataset
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while getting good data ERROR:{e}")
            raise Exception(e)
    

    
    def prepare(self):
        try:
            self.logger.log("info",f"{self.prefix}: started data preparation")
            #feature engineering
            data_prep=dataPreprocessing()
            # get data by removing outliers
            X=self.getGoodData()
            actualData=X
            #Encoding nominal Category
            X=data_prep.encodeCategoricalData(X,0)
            #data cleaning
            X=self.MatchColumnStructure(X)
            X=self.imputeNullValues(X)
            #feature selection
            #X=data_prep.removeConstantFeatures(X)
            #match data structure based on trained model
            #X=self.MatchColumnStructure(X)
            self.logger.log("info",f"{self.prefix}: Ended data preparation")
            return actualData,X
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while preparing data ERROR:{e}")
            raise Exception(e)

    def imputeNullValues(self,X):
        try:
            self.logger.log("info",f"{self.prefix}: started imputing Null values")
            features=X.columns
            imputer=self.LoadPickle(self.ImputeModelPath)
            X=imputer.transform(X)
            X=pd.DataFrame(X,columns=features)
            self.logger.log("info",f"{self.prefix}: Ended imputing Null values")
            #print(X.head())
            return X
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while imputing Null values ERROR:{e}")
            raise Exception(e)
    
    def LoadPickle(self,path):
        try:
            self.logger.log("info",f"{self.prefix}: started loading saved model")
            model=None
            with open(path, 'rb') as file:
                model=pickle.load(file)
            self.logger.log("info",f"{self.prefix}: started loading saved model")
            return model
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while loading saved model at {path} ERROR:{e}")
            raise Exception(e)
        
    def FindCluster(self,x):
        try:
            self.logger.log("info",f"{self.prefix}: started finding cluster for each data")
            clusterModel=self.LoadPickle(self.clusterPath)
            cluster=clusterModel.predict(x)
            self.logger.log("info",f"{self.prefix}: Ended finding cluster for each data")
            return cluster
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while finding cluster for each data ERROR:{e}")
            raise Exception(e)           
    
    def LoadModel(self,cluster):
        try:
            self.logger.log("info",f"{self.prefix}: started Loading model")
            modelPath=f"{self.ModelPath}/{cluster}/model.pkl"
            model=self.LoadPickle(modelPath)
            self.logger.log("info",f"{self.prefix}: Ended Loading model")
            return model
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while Loading model ERROR:{e}")
            raise Exception(e)  
    
    def predict(self,cluster,X):
        try:
            self.logger.log("info",f"{self.prefix}: started data prediction process")
            model=self.LoadModel(cluster)
            result=model.predict(X)
            self.logger.log("info",f"{self.prefix}: Ended data prediction process")
            return result
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while data prediction process ERROR:{e}")
            raise Exception(e)  
    
    def saveResult(self,df):
        try:
            self.logger.log("info",f"{self.prefix}: started saving predicted results")
            df.to_csv(self.BatchTestResultPath,header=True,index=None)
            self.logger.log("info",f"{self.prefix}: Ended saving predicted results")
            return True
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while saving predicted results ERROR:{e}")
            raise Exception(e)  

    def getPredictedResults(self):
        try:
            self.logger.log("info",f"{self.prefix}: started fetching predicted results")
            result=pd.read_csv(self.BatchTestResultPath)
            self.logger.log("info",f"{self.prefix}: Ended fetching predicted results")
            return self.goodDataPath,self.BatchTestResultPath,result
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while fetching predicted results ERROR:{e}")
            raise Exception(e)

    def MatchColumnStructure(self,X):
        try:
            self.logger.log("info",f"{self.prefix}: started preparing data column names based on model")
            temp=pd.DataFrame()
            feature=X.columns
            structure=self.LoadPickle(self.preserveStruct)
            for i in structure:
                if i in feature:
                    temp[i]=X[i]
                else:
                    temp[i]=0
                temp[i]
            self.logger.log("info",f"{self.prefix}: Ended preparing data column names on model")
            return temp
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while preparing data column names based on model ERROR:{e}")
            raise Exception(e) 

    def ChangeDataType(self,X):
        try:
            self.logger.log("info",f"{self.prefix}: started preparing data structure type based on model")
            feature =X.columns
            schema=PredictValidation().getSchema()
            columnDetails=schema["columnname"]
            X=X.astype(columnDetails)
            self.logger.log("info",f"{self.prefix}: Ended preparing data structure type based on model")
            return X
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error while preparing data structure type based on model ERROR:{e}")
            raise Exception(e)


    def predictData(self,X):
        try:
            self.logger.log("info",f"{self.prefix}: started data prediction process")
            X=self.ChangeDataType(X)
            data_prep=dataPreprocessing()
            X=data_prep.encodeCategoricalData(X,0)
            X=self.MatchColumnStructure(X)
            cluster=self.FindCluster(X)
            if len(cluster)==1:
                X["Target"]=self.predict(cluster[0],X)
            result=round(X["Target"][0],2)
            self.logger.log("info",f"{self.prefix}: Ended data prediction process")
            return result
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error during data prediction process ERROR:{e}")
            raise Exception(e)

    def predictBatchData(self):
        try:
            self.logger.log("info",f"{self.prefix}: started batch data prediction process")
            is_valid=self.validate()
            if is_valid:
                actualData,X=self.prepare()
                X["cluster"]=self.FindCluster(X)
                clusters=X["cluster"].unique()
                results=[]
                for i in clusters:
                    temp=X[X["cluster"]==i]
                    temp2=actualData.iloc[X[X["cluster"]==i].index,:].copy(deep=True)
                    temp=temp.drop("cluster",axis=1)
                    temp2["Target"]=self.predict(i,temp)
                    results.append(temp2)
                finalResult=pd.concat(results)
                self.saveResult(finalResult)
                self.logger.log("info",f"{self.prefix}: Ended batch data prediction process")
                return True
            return False
        except Exception as e:
            self.logger.log("error",f"{self.prefix}: Error during batch data prediction process ERROR:{e}")
            raise Exception(e)
    