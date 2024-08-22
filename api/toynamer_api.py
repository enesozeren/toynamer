import os
from contextlib import asynccontextmanager
from http import HTTPStatus
import torch
from fastapi import FastAPI

from src.inference import generate_name

MODEL_PATH = "outputs/train_run_2024-08-22_21-09-19/best.pth"

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load and clean up model on startup and shutdown."""

    toynamer_model = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
    
    # Set the model and tokenizer in the app state
    app.state.toynamer_model = toynamer_model
    print("Welcome! Model loaded successfully!")

    yield

    del app.state.toynamer_model
    print("Goodbye!")


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def root():
    return {"message": "Welcome to the ToyNamer API!"}


@app.post("/name_generator")
async def name_generator(temperature: float):
    generated_name = generate_name(model=app.state.toynamer_model, temperature=temperature)
    return {"generated_name": generated_name}