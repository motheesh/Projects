from TrainingDataValidation.ValidateTrainingData import TrainValidation
from ApplicationLogger.logger import logger
import os
import pandas as pd

class PredictValidation(TrainValidation):
    def __init__(self):
        super(PredictValidation, self).__init__()
        self.TrainingLogPath="./PredictionLog/DataValidationLog"
        self.Trainlogger=logger(self.TrainingLogPath)
        try:
            self.Trainlogger.log("info","DATA VALIDATION: starting Initialization of TrainValidation object")
            self.BatchFilePath="./PredictionBatchDataset"
            self.SchemaPath="./DatasetSchema/TestingDataSchema.json"
            self.GoodDataPath="./PredictionGoodDataset"
            self.BadDataPath="./PredictionBadDataset"
            self.Trainlogger.log("info","DATA VALIDATION: Ending Initialization of TrainValidation object")
        except Exception as e:
            self.Trainlogger.log("error","DATA VALIDATION: error while Initializing TrainValidation object{e}")
            raise Exception(e)


    def ValidatePredictData(self):
            try:
                #read the train data schema
                self.Trainlogger.log("info","DATA VALIDATION: Starting the Predict Validation")
                predict_schema_json=self.getSchema()
                pattern=predict_schema_json["pattern"]
                columns=predict_schema_json["columnname"]
                for fileName in os.listdir(self.BatchFilePath):
                    is_valid=self.fileNameValidation(fileName,pattern)
                    if is_valid:
                        data=pd.read_csv(f"{self.BatchFilePath}/{fileName}")
                        if self.isAllcolumnPresent(data,columns):
                            self.moveGoodData(data,fileName)
                        else:
                            self.moveBadData(data,fileName)
                self.Trainlogger.log("info","DATA VALIDATION: Ending the Predict Validation")
            except Exception as e:
                self.Trainlogger.log("error",f"DATA VALIDATION: error occured while validating Predict Data {e}")
                raise Exception(e)

