import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model_path=os.path.join("artifacts","model.pkl")
            
            preprocessor=load_object(preprocessor_path)
            model=load_object(model_path)
            
            scaled_data=preprocessor.transform(features)
            
            pred=model.predict(scaled_data)
            
            return pred
            
            
        
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,
                 mes:str,
                 dia:int,
                 marca_general:str,
                 colonia:str,
                 alcaldia:str):
        
        self.mes=mes
        self.marca_general=marca_general
        self.colonia=colonia
        self.dia=dia
        self.alcaldia=alcaldia
                
    def get_data_as_dataframe(self):
            try:
                custom_data_input_dict = {
                    'mes':[self.mes],
                    'marca_general':[self.marca_general],
                    'colonia':[self.colonia],
                    'dia':[self.dia],
                    'alcaldia':[self.alcaldia]
                }
                df = pd.DataFrame(custom_data_input_dict)
                logging.info('Dataframe Gathered')
                return df
            except Exception as e:
                logging.info('Exception Occured in prediction pipeline')
                raise CustomException(e,sys)
            

if __name__=="__main__":
    data = CustomData('FEBRERO',
                      14,
                      'NISSAN',
                      'TEPEYAC INSURGENTES',
                      "GUSTAVO A MADERO")


    pred_df = data.get_data_as_dataframe()

    print(pred_df)

    predict_pipeline=PredictPipeline()

    results=predict_pipeline.predict(pred_df)

    print(f'Prediction = {results[0]}')