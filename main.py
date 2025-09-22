# main.py
import os
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from uuid import uuid4

app = FastAPI(title="Cylinder QC Backend")

# Allow CORS for your Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure folders exist
os.makedirs("outputs", exist_ok=True)

@app.post("/predict")
async def predict(
    engineer: str = Form(...),
    sample_id: str = Form(...),
    file: UploadFile = File(...)
):
    # Save uploaded image
    filename = f"outputs/{uuid4().hex}_{file.filename}"
    with open(filename, "wb") as f:
        f.write(await file.read())

    # =========================
    # Dummy analysis (replace with real ML later)
    # =========================
    results = {
        "Cylinders": 3,
        "Voids": 1,
        "Surface Pores": 2,
        "Honeycombing": 0
    }

    # =========================
    # Create dummy PDF
    pdf_path = f"outputs/{uuid4().hex}_report.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16)
    pdf.cell(0, 10, f"Cylinder QC Report", ln=True, align="C")
    pdf.cell(0, 10, f"Inspector: {engineer}", ln=True)
    pdf.cell(0, 10, f"Sample ID: {sample_id}", ln=True)
    for k, v in results.items():
        pdf.cell(0, 10, f"{k}: {v}", ln=True)
    pdf.output(pdf_path)

    # =========================
    # Create dummy annotated image
    annotated_path = f"outputs/{uuid4().hex}_annotated.jpg"
    with Image.open(filename) as im:
        draw = ImageDraw.Draw(im)
        draw.text((10,10), f"Sample: {sample_id}", fill="red")
        im.save(annotated_path)

    # =========================
    # Create dummy CSV
    csv_path = f"outputs/{uuid4().hex}_log.csv"
    df = pd.DataFrame([{"Cylinder": i+1, "Defects": "None"} for i in range(results["Cylinders"])])
    df.to_csv(csv_path, index=False)

    # Return URLs (for Render, serve static files via /outputs/<filename>)
    base_url = "https://<YOUR_RENDER_APP>.onrender.com/outputs"

    response = {
        **results,
        "Report": f"{base_url}/{os.path.basename(pdf_path)}",
        "Annotated": f"{base_url}/{os.path.basename(annotated_path)}",
        "CSV Log": f"{base_url}/{os.path.basename(csv_path)}"
    }
    return JSONResponse(response)

# Serve static files
@app.get("/outputs/{filename}")
async def get_file(filename: str):
    file_path = f"outputs/{filename}"
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "File not found"}
