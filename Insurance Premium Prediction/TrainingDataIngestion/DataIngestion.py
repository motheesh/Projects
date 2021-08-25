from TrainingDataIngestion.TrainDataDetails import TrainDataDetails
from TrainingDataIngestion.dbOperation import dboperations
from ApplicationLogger.logger import logger
import os
import pandas as pd
class DataIngestion:
    def __init__(self):
        self.logPath="./TrainingLog/DataIngestionLog"
        self.logger=logger(self.logPath)
        try:
            self.logger.log("info",f"DATA INGESTION: starting initialization data ingestion")
            self.queryDetails=TrainDataDetails().getQueryDetails()
            self.goodDataPath="./TrainingGoodDataset"
            self.trainingInputPath="./TrainingInputData"
            self.db=dboperations()
            self.logger.log("info",f"DATA INGESTION: Ending initialization data ingestion")
        except Exception as e:
            self.logger.log("error",f"DATA INGESTION: Error while initializing data ingestion ERROR:{e}")
            raise Exception(e)
            
    def getGoodData(self):
        try:
            self.logger.log("info",f"DATA INGESTION: starting get good data")
            df=[]
            for fileName in os.listdir(self.goodDataPath):
                if fileName.__contains__("csv"):
                    temp=pd.read_csv(self.goodDataPath+"/"+fileName)
                    df.append(temp)
            dataset=pd.concat(df)
            self.logger.log("info",f"DATA INGESTION: Ending get good data")
            return dataset
        except Exception as e:
            self.logger.log("error",f"DATA INGESTION: Error while getting good data ERROR:{e}")
            raise Exception(e)
        
    
    def addID(self,df):
        df["id"]=df.index
        return df
    def dropID(self,df):
        df=df.drop("id",axis=1)
        return df
        
    
    def UploadDataToDB(self):
        try:
            self.logger.log("info",f"DATA INGESTION: Starting data upload to DB")
            #get query details 
            queryDetails=self.queryDetails
            #get good data 
            df=self.getGoodData()
            if len(df)>0:
                #initialize DB connection
                db=self.db
                #add id to dataSet
                df=self.addID(df)
                #drop table
                db.dropTable(queryDetails.drop)
                #create table
                db.createTable(queryDetails.create)
                #insert data
                db.insertBatchData(queryDetails.insert,df,queryDetails.tableColumns)
                #close db connection
                db.close()
                self.logger.log("info",f"DATA INGESTION: data uploaded to DB successfully and created Training input csv file")
        except Exception as e:
                self.logger.log("error",f"DATA INGESTION: Error while uploading data to DB ERROR:{e}")
                raise Exception(e)
            
        
    def getDataFromDB(self):
        try:
            self.db=dboperations()
            self.logger.log("info",f"DATA INGESTION: Starting get data from DB")
            result=self.db.getTrainData(self.queryDetails.select)
            self.logger.log("info",f"DATA INGESTION: Ending get data from DB")
            self.db.close()
            df=self.ConvertResultsetToDF(result)
            return df
        except Exception as e:
            self.logger.log("error",f"DATA INGESTION: error while getting data from DB :{e}")
            raise Exception(e)
            
    def ConvertResultsetToDF(self,result):
        try:
            self.logger.log("info",f"DATA INGESTION: Starting convert Rs to DF")
            df=pd.DataFrame(result.all())
            df=self.dropID(df)
            self.logger.log("info",f"DATA INGESTION: Ending convert Rs to DF")
            return df
        except Exception as e:
            self.logger.log("error",f"DATA INGESTION: error while converting Rs to DF{e}")
            raise Exception(e)
            
    
    def CreateInput_CSVFromDB(self):
        try:
            self.logger.log("info",f"DATA INGESTION: Starting create training input")
            df=self.getDataFromDB()
            df.to_csv(f"{self.trainingInputPath}/{self.queryDetails.filename}.csv",header=True,index=None)
            self.logger.log("info",f"DATA INGESTION: Ending create training input")
            return 1
        except Exception as e:
            self.logger.log("error",f"DATA INGESTION: error while converting df to CSV{e}")
            return 0
        