import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor,RandomForestRegressor
from sklearn.linear_model import Ridge,LinearRegression
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split,GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import Lasso,LassoCV
from sklearn.cluster import KMeans
from kneed import KneeLocator
from ApplicationLogger.logger import logger
import os
import pickle

class ModelBuilding:
    def __init__(self):
        self.logPath="./TrainingLog/TrainingModelLog"
        self.logger=logger(self.logPath)
        try:
            self.logger.log("info",f"MODEL TRAINING: starting initialization of model building")
            self.TrainedModelPath="./TrainedModels"
            self.logger.log("info",f"MODEL TRAINING: Ending initialization of model building")
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while initializing model building ERROR:{e}")
            raise Exception(e)

    def getBestEstimator(self,model,parameter,x_train,y_train):
        try:
            self.logger.log("info",f"MODEL TRAINING: started getting best estimator using GridSearch")
            grid=GridSearchCV(model,param_grid=parameter,n_jobs=4)
            grid.fit(x_train,y_train)
            best_estimator=grid.best_estimator_
            self.logger.log("info",f"MODEL TRAINING: Ended getting best estimator using GridSearch")
            return best_estimator
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while getting best estimator using GridSearch ERROR:{e}")
            raise Exception(e)

    def IsModelOverFitting(self,model,x_train,y_train,x_test,y_test):
        try:
            self.logger.log("info",f"MODEL TRAINING: started checking for model overfitting")
            result=None
            train_score=round(model.score(x_train,y_train),2)*100
            test_score=round(model.score(x_test,y_test),2)*100
            isDiff_moreThan5Percent=train_score-test_score
            if isDiff_moreThan5Percent>5:
                result= {"overfitting":True,"train_score":train_score,"test_score":test_score}
            else:
                result= {"overfitting":False,"train_score":train_score,"test_score":test_score}
            self.logger.log("info",f"MODEL TRAINING: Ended checking of model overfitting")
            return result
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while checking model overfitting ERROR:{e}")
            raise Exception(e)
         

    def get_ModelParameters(self,ModelName):
        try:
            self.logger.log("info",f"MODEL TRAINING: started fetching model paramter for {ModelName}")
            parameters={
                "RandomForest":  {
                                'max_depth': [2,3],
                                'criterion':["mse","mae"],
                                'max_features': ['auto','sqrt', 'log2'],
                                'min_samples_leaf': [15,25,35],
                                'min_samples_split': [10,20,30],
                                'n_estimators': [100],
                                'max_samples':[.8],
                                'ccp_alpha':[0.5,10,50,100]
                                },
                "GradientBoost":{
                                'max_depth': [2,3], 
                                'max_features': ['auto', 'log2'],
                                'min_samples_leaf':[25,35,45],
                                'min_samples_split': [20,30,50],
                                'subsample':[0.8],
                                'n_estimators': [100],
                                'ccp_alpha':[30,50,100]     
                                },
                "XgBoost":{"booster":["gbtree"],
                        "objective":['reg:squarederror'],
                        "max_depth":[2],
                        "subsample":[0.8],
                        "reg_lambda":[25,30,50,80],
                        "reg_alpha":[25,30,50,80],
                        "n_estimators":[100],
                        "learning_rate":[0.1],
                        "min_child_weight":[30,40,50],
                        "colsample_bylevel":[0.8],
                        "colsample_bytree":[0.8],
                        "colsample_bynode":[0.8],
                        "gamma":[0.2,0.3]
                        }
                }
            result=None
            if ModelName in parameters:
                result= parameters[ModelName]
            self.logger.log("info",f"MODEL TRAINING: Ended fetching model paramter for {ModelName}")
            return result    
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while fetching model paramter for {ModelName} ERROR:{e}")
            raise Exception(e)
    def getModels(self):
        try:
            self.logger.log("info",f"MODEL TRAINING: started fetching list of models to train ")
            models={
                    #"linear":LinearRegression(),
                    #"Ridge":Ridge(),
                    #"SVM":SVR(),
                    #"XgBoost":XGBRegressor(),
                    #"RandomForest":RandomForestRegressor(),
                    "GradientBoost":GradientBoostingRegressor()
                }
            self.logger.log("info",f"MODEL TRAINING: Ended fetching list of models to train ")
            return models
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while fetching model list to train ERROR:{e}")
            raise Exception(e)
       
    def createModelFolder(self,n_clstr):
        try:
            self.logger.log("info",f"MODEL TRAINING: started creating folder for saving models for each cluster")
            for i in range(0,n_clstr):
                CHECK_FOLDER = os.path.isdir(f"{self.TrainedModelPath}/{i}")
                if not CHECK_FOLDER:
                    os.makedirs(f"{self.TrainedModelPath}/{i}")
            self.logger.log("info",f"MODEL TRAINING: Ended creating folder for saving models for each cluster")
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while creating folder for saving models for each cluster ERROR:{e}")

                
    def UpdateModelDetails(self,path,cluster,model):
        try:
            self.logger.log("info",f"MODEL TRAINING: started updating best model details")
            format_=f'cluster:{cluster} \nModel Name:{model["name"]} \nTraining Score:{model["score_details"]["train_score"]} \nTest Score:{model["score_details"]["test_score"]} \nParamters:{model["model"].get_params()}'
            with open(path, "w") as file:
                file.write(format_)
            self.logger.log("info",f"MODEL TRAINING: Ended updating best model details")
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while updating best model details ERROR:{e}")
            
    def saveModel(self,cluster,model):
        try:
            self.logger.log("info",f"MODEL TRAINING: started saving the best model")
            filename=f"{self.TrainedModelPath}/{cluster}/model.pkl"
            path=f"{self.TrainedModelPath}/{cluster}/modelDetails.txt"
            self.UpdateModelDetails(path,cluster,model)
            pickle.dump(model["model"],open(filename,'wb'))
            self.logger.log("info",f"MODEL TRAINING: Ended saving the best model")
            return True
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while saving the best model ERROR:{e}")
            raise Exception(e)
                
    def TrainAndSaveModel(self,data,n_clstr):
        try:
            self.logger.log("info",f"MODEL TRAINING: started Train and save model")
            Trained_model_details,cluster_wise_best_estimators=self.getBestTrainedModel(data,n_clstr)
            for i in cluster_wise_best_estimators.keys():
                model=cluster_wise_best_estimators[i]
                self.saveModel(i,model)
            self.logger.log("info",f"MODEL TRAINING: Ended Train and save model")
            return True
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while Train and save model ERROR:{e}")
            raise Exception(e)
                
            
    def getBestTrainedModel(self,data,n_clstr):
        try:
            self.logger.log("info",f"MODEL TRAINING: started get best trained model")
            self.createModelFolder(n_clstr)
            models=self.getModels()
            cluster_wise_best_estimators={}
            Trained_model_details=[]
            for i in range(0,n_clstr):
                clstr_x=data[data["clusters"]==i]
                clstr_x=clstr_x.drop(["clusters","Target"],axis=1)
                clstr_y=data[data["clusters"]==i]["Target"]
                x_train,x_test,y_train,y_test=train_test_split(clstr_x,clstr_y,test_size=0.3,random_state=12)
                large=0
                for key,model in zip(models.keys(),models.values()):
                    #getting the hyperparamters for model
                    parameter=self.get_ModelParameters(key)
                    if parameter!=None:
                        best_estimator=self.getBestEstimator(model,parameter,x_train,y_train)
                        scoreDetails=self.IsModelOverFitting(best_estimator,x_train,y_train,x_test,y_test)
                        if not scoreDetails["overfitting"] and scoreDetails["train_score"]>large:
                            cluster_wise_best_estimators[i]={"name":key,"model":best_estimator,
                                                            "score_details":scoreDetails}
                            Trained_model_details.append({"cluster":i,"name":key,
                                                                                "score_details":scoreDetails})
                            large=scoreDetails["train_score"]
                        else:
                            Trained_model_details.append({"cluster":i,"name":key,"score_details":scoreDetails})
                    else:
                        model.fit(x_train,y_train)
                        scoreDetails=self.IsModelOverFitting(model,x_train,y_train,x_test,y_test)
                        if not scoreDetails["overfitting"] and scoreDetails["train_score"]>large:
                            cluster_wise_best_estimators[i]={"name":key,"model":model,"score_details":scoreDetails}
                            Trained_model_details.append({"cluster":i,"name":key,"score_details":scoreDetails})
                            large=scoreDetails["train_score"]
                        else:
                            Trained_model_details.append({"cluster":i,"name":key,"score_details":scoreDetails})
            self.logger.log("info",f"MODEL TRAINING: Ended get best trained model")
            return Trained_model_details,cluster_wise_best_estimators
        except Exception as e:
            self.logger.log("error",f"MODEL TRAINING: Error while getting best trained model ERROR:{e}")
            raise Exception(e)
