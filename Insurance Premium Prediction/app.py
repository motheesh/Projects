from flask import Flask, request, render_template,jsonify,abort,g 
from flask_cors import CORS, cross_origin
from ApplicationLogger.logger import logger
from DataPreparation.dataPreprocessing import dataPreprocessing
import pandas as pd
from DataPrediction.PredictData import DataPrediction
from ApplicationLogger.logger import logger
from TrainingModel.TrainModel import ModelTraining
import time

app=Flask(__name__)
CORS(app)
app.config.from_pyfile("config.py")
logger=logger()
prefix="API CALL: "

@app.before_request
def before_request():
    g.start = round(time.time()* 1000)

@app.route("/")
@app.route("/index")
def home():
    try:
        logger.log("info",f"{prefix}Received request for index page")
        data=dataPreprocessing().GetTemplateDetails()
        logger.log("info",f"{prefix}Responded for index page request")
        return render_template("index.html",data=data)
    except Exception as e:
        logger.log("error",f"{prefix}Error while rendering index page request {e}")
        abort(500)


@app.route("/predict" , methods=['POST'])
def predict():
    try:
        logger.log("info",f"{prefix}Received request for predicting data")
        req_data=request.get_json()
        df=pd.json_normalize(req_data)
        result=DataPrediction().predictData(df)
        responseTime=round(time.time()* 1000)-g.start
        logger.log("info",f"{prefix}Responded to request for predicting data")
        return jsonify({"result":result,"responseTime":responseTime})
    except Exception as e:
        logger.log("error",f"{prefix}Error while predicting data based on request")
        return jsonify({"status":"error","error":e,"message":"Something went wrong please try again"})

@app.route("/predictBatchData" , methods=['GET'])
def predictBatchData():
    try:
        logger.log("info",f"{prefix}Received batch data prediction request")
        result=DataPrediction().predictBatchData()
        if result:
            logger.log("info",f"{prefix}Responded to Batch data prediction request")
            return jsonify({"status":"success","message":"successfuly predicted and saved result"})
    except Exception as e:
        logger.log("error",f"{prefix}Error during data prediction process")
        return jsonify({"status":"error","error":e,"message":"Something went wrong please try again"})

@app.route("/TrainModel" , methods=['GET'])
def TrainModel():
    try:
        logger.log("info",f"{prefix}Received batch data model training request")
        ModelTraining().train()
        logger.log("info",f"{prefix}Responded to batch data model training request")
        return jsonify({"status":"success","message":"successfuly Trained model and ready for prediction"})
    except Exception as e:
        logger.log("error",f"{prefix}Error while Training model process")
        return jsonify({"status":"error","error":e,"message":"Something went wrong please try again"})

@app.route("/getTrainedModelDetails" , methods=['GET'])
def getTrainedModelDetails():
    try:
        logger.log("info",f"{prefix}Received get Trained model details request")
        modelDetails=ModelTraining().getTrainedModelDetails()
        logger.log("info",f"{prefix}Responded to get Trained model details request")
        return render_template("TrainedModelDetails.html",modelDetails=modelDetails)
    except Exception as e:
        logger.log("error",f"{prefix}Error while getting Trained model details")
        return jsonify({"status":"error","error":e,"message":"Something went wrong please try again"})

@app.route("/getPredcitedBatchResults" , methods=['GET'])
def getPredcitedBatchResults():
    try:
        logger.log("info",f"{prefix}Received get batch predicted results request")
        sourcepath,resultpath,results=DataPrediction().getPredictedResults()
        logger.log("info",f"{prefix}Responded to get batch predicted results request")
        return render_template("PredictedResults.html",results=results,resultpath=resultpath,sourcepath=sourcepath)
    except Exception as e:
        logger.log("error",f"{prefix}Error while getting batch predicted results details")
        return jsonify({"status":"error","error":e,"message":"Something went wrong please try again"})

if __name__=="__main__":
    app.run(port=5000) 