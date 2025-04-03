from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
import pandas as pd
import numpy as np
from config.settings import settings

# Initialize router
router = APIRouter()

# Setup templates
templates = Jinja2Templates(directory=str(Path(__file__).parent.parent / "templates"))

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@router.get("/transaction", response_class=HTMLResponse)
async def transaction_page(request: Request):
    return templates.TemplateResponse("transaction.html", {"request": request})

@router.post("/check-fraud", response_class=HTMLResponse)
async def check_fraud(
    request: Request,
    amount: float = Form(...),
    time: float = Form(...),
    v1: float = Form(...),
    v2: float = Form(...),
    v3: float = Form(...),
    v4: float = Form(...),
    v5: float = Form(...),
    v6: float = Form(...),
    v7: float = Form(...),
    v8: float = Form(...),
    v9: float = Form(...),
    v10: float = Form(...),
    v11: float = Form(...),
    v12: float = Form(...),
    v13: float = Form(...),
    v14: float = Form(...),
    v15: float = Form(...),
    v16: float = Form(...),
    v17: float = Form(...),
    v18: float = Form(...),
    v19: float = Form(...),
    v20: float = Form(...),
    v21: float = Form(...),
    v22: float = Form(...),
    v23: float = Form(...),
    v24: float = Form(...),
    v25: float = Form(...),
    v26: float = Form(...),
    v27: float = Form(...),
    v28: float = Form(...)
):
    # Process form data and make prediction
    transaction = {
        "Time": time,
        "Amount": amount,
        "V1": v1, "V2": v2, "V3": v3, "V4": v4, "V5": v5,
        "V6": v6, "V7": v7, "V8": v8, "V9": v9, "V10": v10,
        "V11": v11, "V12": v12, "V13": v13, "V14": v14, "V15": v15,
        "V16": v16, "V17": v17, "V18": v18, "V19": v19, "V20": v20,
        "V21": v21, "V22": v22, "V23": v23, "V24": v24, "V25": v25,
        "V26": v26, "V27": v27, "V28": v28
    }
    
    # Here you would typically call your prediction model
    # For now, we'll return a simple response
    return templates.TemplateResponse("result.html", {
        "request": request,
        "is_fraud": False,  # Replace with actual prediction
        "probability": 0.23  # Replace with actual probability
    })