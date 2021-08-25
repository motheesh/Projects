from TrainingDataValidation.ValidateTrainingData import TrainValidation
from TrainingDataIngestion.DataIngestion import DataIngestion
from DataPreparation.dataPreprocessing import dataPreprocessing
from TrainingModel.ModelBuilding import ModelBuilding
from DataClustering.dataClustering import clusterData
import pandas as pd
from ApplicationLogger.logger import logger
import os


class ModelTraining:
    def __init__(self):
        self.logPath="./TrainingLog/TrainingModelLog"
        self.logger=logger(self.logPath)
        try:
            self.logger.log("info",f"MODEL TRAINING: starting initialization of model Training")
            self.TrainedModelDetailspath="./TrainedModels"
            self.LogPrefix="MODEL TRAINING"
            self.logger.log("info",f"MODEL TRAINING: Ending initialization of model Training")
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while initializing model Training ERROR:{e}")
            raise Exception(e)

    def DataValidation(self):
        try:
            self.logger.log("info",f"{self.LogPrefix}: started data validation")
            #initialize object for TrainValidation
            validate=TrainValidation()
            #validating Training data
            validate.ValidateTrainData()
            self.logger.log("info",f"{self.LogPrefix}: Ended data validation")
        except Exception as e:
            self.logger.log("error",f"{self.LogPrefix}: Error while validating data ERROR:{e}")
            raise Exception(e)
        
    def DataIngestion(self):
        try:
            self.logger.log("info",f"{self.LogPrefix}: started data ingestion")
            DI=DataIngestion()
            #Upload Data
            DI.UploadDataToDB()
            #Conver data to csv for training
            DI.CreateInput_CSVFromDB()
            self.logger.log("info",f"{self.LogPrefix}: Ended data ingestion")
        except Exception as e:
            self.logger.log("error",f"{self.LogPrefix}: Error during data ingestion ERROR:{e}")
            raise Exception(e)
     
    def prepareData(self):
        try:
            self.logger.log("info",f"{self.LogPrefix}: started data preparation")
            #feature engineering
            data_prep=dataPreprocessing()
            # get data by removing outliers
            X,Y=data_prep.getTrainingData()
            #Encoding nominal Category
            X=data_prep.encodeCategoricalData(X)
            #data cleaning
            X=data_prep.imputeNullValues(X)
            #feature selection
            X=data_prep.removeConstantFeatures(X)
            data_prep.preserveColumnNamesForPrediction(X)
            self.logger.log("info",f"{self.LogPrefix}: Ended data preparation")
            return X,Y
        except Exception as e:
            self.logger.log("error",f"{self.LogPrefix}: Error while preparing data ERROR:{e}")
            raise Exception(e)
    
    def ClusterData(self,X,Y):
        try:
            self.logger.log("info",f"{self.LogPrefix}: started data clustering")
            #data Clustering - clustering data based on similarities and train each cluster with different models
            cluster=clusterData()
            data,n_clstr=cluster.ClusterData(X,Y)
            self.logger.log("info",f"{self.LogPrefix}: Ended data clustering")
            return data,n_clstr
        except Exception as e:
            self.logger.log("error",f"{self.LogPrefix}: Error while clustering data ERROR:{e}")
            raise Exception(e)
    
    def BuilModel(self,data,n_clstr):
        try:
            self.logger.log("info",f"{self.LogPrefix}: starting model building")
            trainModel=ModelBuilding()
            #train and get the save the best model for each cluster
            trainModel.TrainAndSaveModel(data,n_clstr)
            self.logger.log("info",f"{self.LogPrefix}: Ended model building")
        except Exception as e:
            self.logger.log("error",f"{self.LogPrefix}: Error while building model ERROR:{e}")
            raise Exception(e)

    
    def TrainModel(self):
        try:
            self.logger.log("info",f"{self.LogPrefix}: started model Training")
            #Prepare the data for model bulding
            X,Y=self.prepareData()
            #cluster the data and {self.LogPrefix} on each cluster
            data,n_clstr=self.ClusterData(X,Y)
            #build and save the model with high accuracy
            self.BuilModel(data,n_clstr)
            self.logger.log("info",f"{self.LogPrefix}: Ended model Training")
        except Exception as e:
            self.logger.log("error",f"{self.LogPrefix}: Error while training model ERROR:{e}")
            raise Exception(e)

    def train(self):
        try:
            self.logger.log("info",f"{self.LogPrefix}: started Training process")
            #Validate data from client source based on pre-defined dataset schema
            self.DataValidation()
            #upload valid data to DB and export to csv for training
            self.DataIngestion()
            #Train the model and save it for prediction
            self.TrainModel()
            self.logger.log("info",f"{self.LogPrefix}: Ended Training process")
        except Exception as e:
            self.logger.log("error",f"{self.LogPrefix}: Error during training process ERROR:{e}")
            raise Exception(e)

    def getTrainedModelDetails(self):
        try:
            self.logger.log("info",f"{self.LogPrefix}: started getting trained model Details")
            ModelDetails=[]
            path=self.TrainedModelDetailspath
            directory_list = list()
            for root, dirs, files in os.walk(path, topdown=False):
                for name in dirs:
                    if "." not in name:
                        directory_list.append(f"{path}/{name}/modelDetails.txt")
            for i in directory_list:
                if os.path.exists(i):
                    with open(i,"r") as file:
                        ModelDetails.append(file.read())
            self.logger.log("info",f"{self.LogPrefix}: Ended getting trained model Details")
            return ModelDetails
        except Exception as e:
            self.logger.log("error",f"{self.LogPrefix}: Error while getting trained model DetailsERROR:{e}")
            raise Exception(e)