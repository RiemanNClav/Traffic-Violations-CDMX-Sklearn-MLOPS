import os
import sys
import pandas as pd
from src.logger import logging
from src.exception import CustomException

from dataclasses import dataclass, field


from src.utils import categoria_mas_parecida






@dataclass
class DataUnionConfig:
    raw_data_path: str=os.path.join('notebook','data',"data.csv")

    columns: list = field(default_factory=lambda: ['id_infraccion', 'fecha_infraccion', 'ao_infraccion', 'mes', 'categoria', 
                    'articulo', 'fraccion', 'inciso', 'parrafo', 'placa', 
                    'Color', 'marca_general', 'marca', 'submarca', 'en_la_calle', 
                    'entre_calle', 'y_calle', 'colonia', 'alcaldia', 'longitud', 'latitud'])

class DataUnion:
    def __init__(self):
        self.ingestion_config=DataUnionConfig()

    def initiate_data_union(self):
        logging.info("Initializated data union")
        try:

            file_path = os.path.join('notebook','data',"corruptos.txt")
            os.remove(file_path)

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            directorio = 'notebook/data/'
            palabra_clave = 'infracciones_infracciones_transito'
            dataframes = []
            str = ''
            for archivo in os.listdir(directorio):
                if archivo.endswith('.csv') and palabra_clave in archivo:
                    path = os.path.join(directorio, archivo)
                    df = pd.read_csv(path, encoding="utf-8")

                    try:

                        if list(df.columns) == self.ingestion_config.columns:

                            df = df[self.ingestion_config.columns]

                        else:
                            new_names = {col: categoria_mas_parecida(col, self.ingestion_config.columns) for col in df.columns}
                            for old_name, new_name in new_names.items():
                                df = df.rename(columns={old_name: new_name})

                            df = df[self.ingestion_config.columns]

                        df = df.astype({
                            "id_infraccion": "str",
                            "mes": "str",
                            "categoria": "str",
                            "articulo": "str",
                            "fraccion": "str",
                            "inciso": "str",
                            "parrafo": "str",
                            "placa": "str",
                            "Color": "str",
                            "marca_general": "str",
                            "submarca": "str"})\
                            .rename(columns={"en_la_calle": "calle1",
                                             "entre_calle": "calle2",
                                             "y_calle": "calle3",
                                             "ao_infraccion": "anio_infraccion",
                                             "Color": "Color",
                                             "alcaldia": "alcaldia",
                                             "colonia": "colonia"})
                        
                        df.to_csv(self.ingestion_config.raw_data_path, index=False)
                        str = 'NO HAY'
                        os.remove(path)
                        
                    except:
                        str += archivo + '\n'

            with open(file_path, 'w') as file:
                file.write(str)

            logging.info("Finished data union")

            return(df)
        
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataUnion()
    obj.initiate_data_union()