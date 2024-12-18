import pandas as pd
import numpy as np
import mlflow
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Set up MLflow
mlflow.set_tracking_uri('https://dagshub.com/WajihHlaili/my-first-repo.mlflow')

# Load the model once at the start
df_mlflow = mlflow.search_runs(filter_string="metrics.Accuracy<1")
run_id = df_mlflow.loc[df_mlflow['metrics.Accuracy'].idxmax()]['run_id']
logged_model = f'runs:/{run_id}/ML_models'
model = mlflow.pyfunc.load_model(logged_model)

# FastAPI app setup
app = FastAPI()

# CORS middleware to allow requests from all origins (can be more restrictive in production)
origins = ['*']
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Item(BaseModel):
    transmission: int
    fuel: int
    owner: int
    year: int
    km_driven: int
    engine: int
    max_power: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Car Price Prediction API"}

@app.post("/predict")
def predict(item: Item):
    print(f"Received data: {item}")  # Debug log to check received data

    # Prepare the input data for prediction
    data = np.array([
        item.transmission,
        item.fuel,
        item.owner,
        item.year,
        item.km_driven,
        item.engine,
        item.max_power
    ]).reshape(1, -1)

    # Perform prediction
    prediction = model.predict(data.reshape(1, -1))

    print(f"Prediction: {prediction[0]}")  # Debug log to check prediction result

    # Return the prediction result
    return {"Predict": prediction[0]}



# Run the FastAPI app
import uvicorn
if __name__ == "__main__":
    import asyncio

    async def main():
        config = uvicorn.Config(app, host="0.0.0.0", port=8000)
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main()) 