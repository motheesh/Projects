#imports
import numpy as np
import pandas as pd
import json
import os
import re
from ApplicationLogger.logger import logger

class TrainValidation:
    def __init__(self):
        self.TrainingLogPath="./TrainingLog/DataValidationLog"
        self.Trainlogger=logger(self.TrainingLogPath)
        try:
            self.Trainlogger.log("info","DATA VALIDATION: starting Initialization of Validation object")
            self.BatchFilePath="./TrainingBatchDataset"
            self.SchemaPath="./DatasetSchema/TrainingDataSchema.json"
            self.GoodDataPath="./TrainingGoodDataset"
            self.BadDataPath="./TrainingBadDataset"
            self.Trainlogger.log("info","DATA VALIDATION: Ending Initialization of Validation object")
        except Exception as e:
            self.Trainlogger.log("error","DATA VALIDATION: error while Initializing Validation object{e}")
            raise Exception(e)
            

        
        
    def getSchema(self):
        try:
            self.Trainlogger.log("info","DATA VALIDATION: Started reading TrainSchema json")
            f = open (self.SchemaPath, "r")
            train_schema_json=json.loads(f.read())
            f.close()
            self.Trainlogger.log("info","DATA VALIDATION: Ended reading TrainSchema json")
            return train_schema_json
        except Exception as e:
            self.Trainlogger.log("error","DATA VALIDATION: Error occured while reading training schema")
            raise Exception(e)

        
    
    def fileNameValidation(self,file,pattern):
        result=False
        try:
            self.Trainlogger.log("info","DATA VALIDATION: starting file name validation")
            if re.match(pattern,file):
                result=True
            else:
                self.Trainlogger.log("info","DATA VALIDATION: File name is not matched")
                result=False
            self.Trainlogger.log("info","DATA VALIDATION: ending file name validation")
        except Exception as e:
            self.Trainlogger.log("error",f"DATA VALIDATION: Error occured while validating file name {e}")
        return result
        
    def checkColumnType(self,actualTypes,requiredTypes):
        try:
            self.Trainlogger.log("info","DATA VALIDATION: starting column Type validation")
            result=sum([i in j for i,j in zip(requiredTypes,actualTypes)])==len(requiredTypes)
            self.Trainlogger.log("info","DATA VALIDATION: Ending column Type validation")
            return result
        except Exception as e:
            self.Trainlogger.log("error",f"DATA VALIDATION: Error occured while validating column Types {e}")
            return False
    
    def checkColumnNames(self,actualColumnName,columnNames):
        try:
            self.Trainlogger.log("info","DATA VALIDATION: starting column Names validation")
            result=sum(columnNames==actualColumnName)==len(columnNames)
            self.Trainlogger.log("info","DATA VALIDATION: Ending column Names validation")
            return result
        except Exception as e:
            self.Trainlogger.log("error",f"DATA VALIDATION: Error occured while validating column Names {e}")
            return False
    
    def allCol_LessThan95_NA(self,data):
        try:
            self.Trainlogger.log("info","DATA VALIDATION: starting >95% Na validation")
            dd=(data.isna().sum()/len(data))*100
            result=True
            for i in dd.index:
                if dd[i]>95:
                    self.Trainlogger.log("info",f"DATA VALIDATION: column {i} have >95% NA so we reject this dataset")
                    result=False
                    break
        except Exception as e:
            self.Trainlogger.log("error",f"DATA VALIDATION: Error occured while validating allCol_LessThan95_NA {e}")  
        self.Trainlogger.log("info",f"DATA VALIDATION: completing >95% NA validation")
        return result
    
    def isAllcolumnPresent(self,data,columns):
        try:
            self.Trainlogger.log("info","DATA VALIDATION: Starting column values and data type check")
            columnLength,actualLength=len(columns),len(data.columns)
            actualColumnName,actualColumnType=data.dtypes.index,data.dtypes.values.astype("str")
            columnNames,columnTypes=list(columns.keys()),columns.values()
            if actualLength==columnLength:
                if self.checkColumnNames(actualColumnName,columnNames):
                    if self.checkColumnType(actualColumnType,columnTypes):
                        if self.allCol_LessThan95_NA(data):
                            self.Trainlogger.log("info","DATA VALIDATION: Column validation succesfull")
                            return True
                        else:
                            return False
                else:
                    return False
            else:
                return False
            self.Trainlogger.log("info","DATA VALIDATION: Ended column values and data type check")
        except Exception as e:
            self.Trainlogger.log("error",f"DATA VALIDATION: error occured during column validation {e}")
            return False
            
    
    def replace_NaWithNULL(self,data):
        try:
            self.Trainlogger.log("info","DATA VALIDATION: started Replacing NA values with NULL")
            data=data.fillna("NULL")
            self.Trainlogger.log("info","DATA VALIDATION: ended Replacing NA values with NULL")
        except Exception as e:
            self.Trainlogger.log("error",f"DATA VALIDATION: error while Replacing NA values with NULL {e}")
        return data
    
    def moveGoodData(self,data,filename):
        try:
            self.Trainlogger.log("info","DATA VALIDATION: started data moving to good data folder")
            data.to_csv(f"{self.GoodDataPath}/{filename}",header=True,index=None)
            self.Trainlogger.log("info","DATA VALIDATION: ending data movement to good data folder")
            return True
        except Exception as e:
            self.Trainlogger.log("error",f"DATA VALIDATION: error occured while moving file to good data folder {e}")
            return False
            
    
    def moveBadData(self,data,filename):
        try:
            self.Trainlogger.log("info","DATA VALIDATION: started data moving to bad data folder")
            data.to_csv(f"{self.BadDataPath}/{filename}",header=True,index=None)
            self.Trainlogger.log("info","DATA VALIDATION: ending data movement to bad data folder")
            return True
        except Exception as e:
            self.Trainlogger.log("error",f"DATA VALIDATION: error occured while moving file to bad data folder {e}")
            return False
    
    def ValidateTrainData(self):
            try:
                #read the train data schema
                self.Trainlogger.log("info","DATA VALIDATION: Starting the Train Validation")
                train_schema_json=self.getSchema()
                pattern=train_schema_json["pattern"]
                columns=train_schema_json["columnname"]
                for fileName in os.listdir(self.BatchFilePath):
                    is_valid=self.fileNameValidation(fileName,pattern)
                    if is_valid:
                        data=pd.read_csv(f"{self.BatchFilePath}/{fileName}")
                        if self.isAllcolumnPresent(data,columns):
                            data=self.replace_NaWithNULL(data)
                            self.moveGoodData(data,fileName)
                        else:
                            self.moveBadData(data,fileName)
                self.Trainlogger.log("info","DATA VALIDATION: Ending the Train Validation")
            except Exception as e:
                self.Trainlogger.log("error",f"DATA VALIDATION: error occured while validating Train Data {e}")
                raise Exception(e)


