from operator import index
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import sklearn
from fastapi import FastAPI, File, UploadFile,Request
import uvicorn
import sys  
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import mlflow
import os
import asyncio
import uvicorn


# # 1. Configuration de daghubs


os.environ['MLFLOW_TRACKING_USERNAME']= "WajihHlaili"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "WajihHlaili888"

#setup mlflow
mlflow.set_tracking_uri('https://dagshub.com/WajihHlaili/my-first-repo.mlflow')

app = FastAPI()
origins = ['*']

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


#let's call the model from the model registry ( in production stage)

df_mlflow = mlflow.search_runs(filter_string="metrics.Accuracy<1")
run_id = df_mlflow.loc[df_mlflow['metrics.Accuracy'].idxmax()]['run_id']

logged_model = f'runs:/{run_id}/ML_models'

model = mlflow.pyfunc.load_model(logged_model)

# # 2. Launch FastApi

from pydantic import BaseModel, validator
class Item(BaseModel):
    transmission: int
    fuel: int
    owner: int
    year:int
    km_driven:int
    engine:int
    max_power:int

from pydantic import BaseModel, validator
import numpy

@app.get("/")
def read_root():
    return {"Hello": "to Car price prediction app"}


# this endpoint receives data in the form of json (informations about one transaction)
@app.post("/predict")
def predict(item: Item):
    print(item)
    data = [item.transmission, item.fuel, item.owner, item.year, item.km_driven, item.engine, item.max_power]
    data = np.array(data).reshape(1, -1)
    predictions = model.predict(data.reshape(1, -1))
    print(predictions)
    return {"Predict":predictions[0]}


if __name__ == "__main__":
    import asyncio

    async def main():
        config = uvicorn.Config(app, host="127.0.0.1", port=8000)
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())




