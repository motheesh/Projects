import numpy as np
from ApplicationLogger.logger import logger
from TrainingDataValidation.ValidateTrainingData import TrainValidation
import pandas as pd
from ApplicationLogger.logger import logger

class TrainDataDetails:
    def __init__(self):
        self.TrainingLogPath="./TrainingLog/DataIngestionLog"
        self.TrainDbLogger=logger(self.TrainingLogPath)
        try:
            self.TrainDbLogger.log("info","DATA DETAILS: starting initialization of Train data details")
            schema=TrainValidation().getSchema()
            self.columns=self.dict_to_array(schema["columnname"].keys())
            self.tablename=schema["tablename"]
            self.filename=schema["filename"]
            self.datatypes=self.dict_to_array(schema["columnname"].values())
            self.TrainDbLogger.log("info","DATA DETAILS: Ending Initialization of Train data details")
        except Exception as e:
            self.TrainDbLogger.log("error",f"DATA DETAILS: Error while Initializing Train data details {e}")
            

    def getQueryDetails(self):
        try:
            self.TrainDbLogger.log("info",f"DATA DETAILS: Starting preparation of query details")
            self.tableColumns=np.concatenate([["id"],self.columns])
            self.insert=f"insert into insurance.{self.tablename} ({(',').join(self.tableColumns)}) values ({(',').join(['%s' for i in range(len(self.tableColumns))])})"
            self.select=f"select {(',').join([i for i in self.tableColumns])} from insurance.{self.tablename}"
            self.truncate=f"truncate insurance.{self.tablename}"
            self.create=f"create table IF NOT EXISTS insurance.{self.tablename} ({(',').join([f'{i} int primary key' if i=='id' else f'{i} text' for i in self.tableColumns])})"
            self.drop=f"drop table IF EXISTS insurance.{self.tablename}"
            #print(self.insert,self.select,self.truncate,self.create)
            self.TrainDbLogger.log("info",f"DATA DETAILS: Ending preparation of query details")
        except Exception as e:
            self.TrainDbLogger.log("error",f"DATA DETAILS: error during preparing query details {e}")
            
        return self
    def as_dict(self):
        return {"columns":self.columns,"tablename":self.tablename,"filename":self.filename,"datatypes":self.datatypes}
    
    def dict_to_array(self,dictList):
        return np.array(list(dictList))