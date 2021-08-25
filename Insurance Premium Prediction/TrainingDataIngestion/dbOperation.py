from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from cassandra.query import SimpleStatement, BatchStatement
from cassandra.policies import RetryPolicy
import numpy as np
import config
from ApplicationLogger.logger import logger
from TrainingDataValidation.ValidateTrainingData import TrainValidation
import pandas as pd   
    
class dboperations:
    def __init__(self):
        self.TrainingLogPath="./TrainingLog/DataIngestionLog"
        self.TrainDbLogger=logger(self.TrainingLogPath)
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: starting initialization DB Operation")
            self.session=self.CreateSession()
            self.TrainDbLogger.log("info",f"DB OPERATION: Ending initialization DB Operation")
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error while initializing db operation {e}")
            raise Exception(e)
    def CreateSession(self):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: creating DB connection starts ")
            cloud_config= {
                'secure_connect_bundle': config.BUNDLE_PATH
            }
            auth_provider = PlainTextAuthProvider(config.CLIENT_ID, config.SECRET_KEY)
            cluster = Cluster(cloud=cloud_config, auth_provider=auth_provider,protocol_version=4)
            self.cluster=cluster
            session = cluster.connect()
            self.TrainDbLogger.log("info",f"DB OPERATION: created DB connection")
            return session
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error during db connection {e}")
            raise Exception(e)
            
    def close(self):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: started close DB connection operation")
            self.cluster.shutdown()
            self.TrainDbLogger.log("info",f"DB OPERATION: closed DB connection successfully")
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error while closing db connection {e}")
            
    def executeQuery(self,query,values=[]):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: executing query starts")
            session=self.session
            if len(values)>0:
                query=self.prepareQuery(session,query,values)
            result=session.execute(query)
            self.TrainDbLogger.log("info",f"DB OPERATION: Ending query execution")
            return result
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error during query execution {e}")
            raise Exception(e)

    def executeQueryOne(self,query,values=[]):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: starting one query execution")
            session=self.session
            if len(values)>0:
                query=self.prepareQuery(session,query,values)
            result=session.execute(query).one()
            self.TrainDbLogger.log("info",f"DB OPERATION: Ending one query execution")
            return result
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error during query execution {e}")
            raise Exception(e)
            
    def truncateTable(self,query):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: starting Truncate table Operation")
            result=self.executeQueryOne(query)
            self.TrainDbLogger.log("info",f"DB OPERATION: Ending Truncate table Operation")
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error during Truncate table {e}")
            raise Exception(e)
    
    def createTable(self,query):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: starting  create table Operation")
            result=self.executeQueryOne(query)
            self.TrainDbLogger.log("info",f"DB OPERATION: Ending create table Operation")
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error during creating table {e}")
            raise Exception(e)
        
    def dropTable(self,query):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: starting drop table Operation")
            result=self.executeQuery(query)
            self.TrainDbLogger.log("info",f"DB OPERATION: Ending drop table Operation")
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error during dropping table {e}")
            raise Exception(e)
    
    def getTrainData(self,query):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: starting get train data Operation")
            result=self.executeQuery(query)
            self.TrainDbLogger.log("info",f"DB OPERATION: Ending get train data Operation")
            return result
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error during getting data from table using select query {e}")
            raise Exception(e)
        
    def prepareBatchData(self,query,batchList,columns):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: starting Batch Data query preperation")
            batch=BatchStatement()
            for i in range(0,len(batchList)):
                values=tuple([batchList.loc[i][j] if j=="id" else str(batchList.loc[i][j])  for j in columns])
                #print(values)
                #print(query)
                batch.add(SimpleStatement(query),values )
            self.TrainDbLogger.log("info","DB OPERATION: Ending Batch Data query preperation")
            return batch
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: Error while preparing batch insert query {e}")
            raise Exception(e)
        
    def insertBatchData(self,query,batchList,columns):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: starting Batch data insertion")
            session=self.session
            batch=self.prepareBatchData(query,batchList,columns)
            session.execute(batch)
            self.TrainDbLogger.log("info",f"DB OPERATION: Ending Batch data insertion")
            return 1
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error during batch data insertion {e}")
            raise Exception(e)

    def prepareQuery(self,session,query,values):
        try:
            self.TrainDbLogger.log("info",f"DB OPERATION: starting query preperation for execution")
            stmt=session.prepare(query)
            qry=stmt.bind(values)
            self.TrainDbLogger.log("info",f"DB OPERATION: Ending query preperation for execution")
            return qry
        except Exception as e:
            self.TrainDbLogger.log("error",f"DB OPERATION: error during query preparation {e}")
            raise Exception(e)