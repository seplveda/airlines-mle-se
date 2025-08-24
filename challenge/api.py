import fastapi
import pandas as pd
from fastapi import HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator
from typing import List
from challenge.model import DelayModel

app = fastapi.FastAPI()

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={"detail": "Validation error"},
    )

# Initialize and train the model
model = DelayModel()

# Load data and train model once at startup
try:
    data = pd.read_csv("data/data.csv", low_memory=False)
    features, target = model.preprocess(data, target_column="delay")
    model.fit(features, target)
except Exception as e:
    print(f"Warning: Could not train model at startup: {e}")

class Flight(BaseModel):
    OPERA: str
    TIPOVUELO: str
    MES: int
    
    @field_validator('MES')
    def validate_mes(cls, v):
        if v < 1 or v > 12:
            raise ValueError('MES must be between 1 and 12')
        return v
    
    @field_validator('TIPOVUELO')
    def validate_tipovuelo(cls, v):
        if v not in ['I', 'N']:
            raise ValueError('TIPOVUELO must be I or N')
        return v

class FlightRequest(BaseModel):
    flights: List[Flight]

@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }

@app.post("/predict", status_code=200)
async def post_predict(request: FlightRequest) -> dict:
    try:
        # Convert flight data to DataFrame format expected by model
        flights_data = []
        for flight in request.flights:
            flights_data.append({
                'OPERA': flight.OPERA,
                'TIPOVUELO': flight.TIPOVUELO,
                'MES': flight.MES
            })
        
        flights_df = pd.DataFrame(flights_data)
        
        # Add required columns with dummy dates for preprocessing
        flights_df['Fecha-I'] = '2017-01-01 12:00:00'  # Dummy date for feature engineering
        
        # Preprocess the data
        features = model.preprocess(flights_df)
        
        # Make predictions
        predictions = model.predict(features)
        
        return {"predict": predictions}
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")