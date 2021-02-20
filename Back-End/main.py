import pickle
import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel

app = FastAPI()
with open("model.mo", "rb") as f:
    model = pickle.load(f)


class Diagnose(BaseModel):
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
    diagnose_dic = jsonable_encoder(diagnose)
    for key, value in diagnose_dic.items():
        diagnose_dic[key] = [value]

    single_instance = pd.DataFrame.from_dict(diagnose_dic)
    prediction = model.predict(single_instance)
    return prediction[0]
