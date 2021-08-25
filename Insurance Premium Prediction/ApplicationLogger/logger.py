import logging as lg
import time
import logging.config
from os import path

class logger:
    def __init__(self,path="./ApplicationLog"):
        self.logPath=path
        self.log_file_path="./ApplicationLogger/logging.config"
        
    def log(self,errorType,error):
        filename=time.strftime("%m_%d_%Y")
        filePath=f'{self.logPath}/{filename}.log'
        lg.config.fileConfig(self.log_file_path,
                          defaults={'logfilename': filePath},disable_existing_loggers=False)
        #lg.basicConfig(filename=f'{self.logPath}/{filename}.log',level=lg.INFO,
        #               format="%(asctime)s %(name)s %(levelname)s %(message)s")
        #logger = logging.getLogger(__name__)
        if errorType=="error":
            lg.error(error)
        elif errorType=="warning":
            lg.warning(error)
        elif errorType=="debug":
            lg.debug(error)
        elif errorType=="info":
            lg.info(error)
        lg.shutdown()