
import pandas as pd
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from keras.models import model_from_json
from keras.models import load_model


app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:4200",
    "http://127.0.0.1:8000/"
    "http://localhost:8000/"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

json_file = open('model_num.json', 'r')


loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# load weights into new model
model.load_weights("model_num.h5")
print("Loaded model from disk")

model.save('model_num.hdf5')
model = load_model('model_num.hdf5')


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
    print(diagnose)
    diagnose_dict = jsonable_encoder(diagnose)
    diagnose_dict = {k: [v] for (k, v) in jsonable_encoder(diagnose).items()}
    df = pd.DataFrame.from_dict(diagnose_dict)

    # Drop Not Final Independent Variables for Model Due to VIF Calculation
    df.drop(columns=['age', 'evaluation_bi_rads', 'mass_shape'], inplace=True)
    # Predict  Diagnose
    prediction = model.predict_classes(df)

    return prediction[0][0].item()
