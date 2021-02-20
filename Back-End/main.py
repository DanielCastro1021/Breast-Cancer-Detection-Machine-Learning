
import pickle
import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel as bm
from keras.models import model_from_json
from keras.models import load_model


app = FastAPI()

json_file = open('model_num.json', 'r')


loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model_num.h5")
print("Loaded model from disk")

model.save('model_num.hdf5')
model = load_model('model_num.hdf5')


class Diagnose(bm):
    age: int
    evaluation_bi_rads: int
    mass_shape: int
    mass_margin: int
    mass_density: int


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.post("/diagnose_predict")
async def predict_diagnose(diagnose: Diagnose):
    diagnose_dict = jsonable_encoder(diagnose)
    diagnose_dict = {k: [v] for (k, v) in jsonable_encoder(diagnose).items()}
    df = pd.DataFrame.from_dict(diagnose_dict)
    # Drop Not Final Independent Variables for Model Due to VIF Calculation

    df.drop(columns=['age', 'evaluation_bi_rads', 'mass_shape'], inplace=True)
    prediction = model.predict_classes(df)

    return prediction[0][0].item()
