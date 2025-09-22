import os
os.environ["YOLO_CONFIG_DIR"] = "/tmp/Ultralytics"
import os
import gdown
import cv2, torch, numpy as np
from ultralytics import YOLO
from torchvision.ops import nms
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from PIL import Image as PILImage
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import FileResponse, JSONResponse
import shutil

# ---------------- DOWNLOAD YOLO MODELS ----------------
os.makedirs("Weights", exist_ok=True)

weights = {
    "best(2).pt": "1FmHHcjzOfIf2oDe77IeR7Bu2uewYtQG2",
    "best(6).pt": "15XsZVMH6pGMXj3I6J9_Ra1EvvNHC2a8o"
}

for name, file_id in weights.items():
    path = os.path.join("Weights", name)
    if not os.path.exists(path):
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, path, quiet=False)

# ---------------- CONFIG ----------------
C = {
    "M": [os.path.join("Weights", n) for n in weights.keys()],  # YOLO model paths
    "S": 4.8,  # ArUco marker size in cm
    "C": {
        "Cylinder": (255, 0, 0),
        "Surface Pores": (0, 255, 0),
        "Voids": (0, 0, 255),
        "honey_combing": (0, 255, 255)
    },
    "R": "cylinder_report.pdf",
    "O": "ensemble_result.jpg"
}

# ---------------- LOAD YOLO MODELS ----------------
M = [YOLO(p) for p in C["M"]]
N = M[0].names

# ---------------- FUNCTIONS ----------------
def A(img):
    """Pixel-to-inch conversion using ArUco marker."""
    from cv2 import aruco
    aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters()
    detector = aruco.ArucoDetector(aruco_dict, parameters)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, rejected = detector.detectMarkers(gray)
    if len(corners) > 0:
        pixel_dist = np.linalg.norm(corners[0][0][0] - corners[0][0][1])
        return (pixel_dist / C["S"]) * 2.54
    return None

def B(f):
    """YOLO ensemble detection with NMS."""
    r1, r2 = M[0](f)[0], M[1](f)[0]
    b = np.vstack([r1.boxes.xyxy.cpu().numpy(), r2.boxes.xyxy.cpu().numpy()])
    s = np.hstack([r1.boxes.conf.cpu().numpy(), r2.boxes.conf.cpu().numpy()])
    cl = np.hstack([r1.boxes.cls.cpu().numpy(), r2.boxes.cls.cpu().numpy()])
    i = nms(torch.tensor(b), torch.tensor(s), 0.5)
    return [b[j] for j in i], [s[j] for j in i], [cl[j] for j in i]

def C0(f):
    """Analyze image and annotate defects."""
    im = cv2.imread(f)
    px = A(im)
    bx, sc, cl = B(f)
    d = {"file": f, "c": 0, "v": 0, "p": 0, "h": 0, "dims": []}
    for b, s, c in zip(bx, sc, cl):
        x1, y1, x2, y2 = map(int, b)
        nm = N[int(c)]
        col = C["C"].get(nm, (255, 255, 255))
        cv2.rectangle(im, (x1, y1), (x2, y2), col, 2)
        cv2.putText(im, f"{nm} {s:.2f}", (x1, y1 - 10), 0, 0.6, col, 2)
        if nm == "Cylinder":
            d["c"] += 1
            if px:
                h, w = (y2 - y1) / px, (x2 - x1) / px
                d["dims"].append((h, w))
                cv2.putText(im, f"H:{h:.2f}in D:{w:.2f}in", (x1, y2 + 20), 0, 0.6, (255, 0, 0), 2)
        elif nm == "Voids": d["v"] += 1
        elif nm == "Surface Pores": d["p"] += 1
        elif nm == "honey_combing": d["h"] += 1
    cv2.imwrite(C["O"], im)
    return d

def R0(d, engineer="Engineer", sample_id="Sample-001"):
    """Generate PDF report."""
    S = getSampleStyleSheet()
    doc = SimpleDocTemplate(C["R"], pagesize=A4)
    st_pdf = []
    st_pdf.append(Paragraph("<b><font size=18>Technical Laboratory Inspection Report</font></b>", S["Title"]))
    st_pdf.append(Spacer(1, 20))
    st_pdf.append(Paragraph(f"<b>Inspection Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", S["Normal"]))
    st_pdf.append(Paragraph(f"<b>Engineer:</b> {engineer}", S["Normal"]))
    st_pdf.append(Paragraph(f"<b>Sample ID:</b> {sample_id}", S["Normal"]))
    st_pdf.append(Paragraph(f"<b>Tested Image:</b> {os.path.basename(d['file'])}", S["Normal"]))
    st_pdf.append(Spacer(1, 20))
    if d["dims"]:
        for i, (h, w) in enumerate(d["dims"]):
            st_pdf.append(Paragraph(f"<b>Cylinder {i+1}:</b> Height = {h:.2f} in, Diameter = {w:.2f} in", S["Normal"]))
    else:
        st_pdf.append(Paragraph("<b>No measurable cylinders detected.</b>", S["Normal"]))
    st_pdf.append(Spacer(1, 20))
    data = [
        ["Defect Type", "Count"],
        ["Cylinders", d["c"]],
        ["Voids", d["v"]],
        ["Surface Pores", d["p"]],
        ["Honeycombing", d["h"]],
    ]
    table = Table(data, colWidths=[200, 100])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,0), colors.grey),
        ("TEXTCOLOR", (0,0), (-1,0), colors.whitesmoke),
        ("ALIGN", (0,0), (-1,-1), "CENTER"),
        ("FONTNAME", (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE", (0,0), (-1,0), 12),
        ("BOTTOMPADDING", (0,0), (-1,0), 10),
        ("GRID", (0,0), (-1,-1), 1, colors.black),
    ]))
    st_pdf.append(table)
    st_pdf.append(Spacer(1, 20))
    img = PILImage.open(C["O"])
    iw, ih = img.size
    max_w, max_h = 400, 400
    scale = min(max_w / iw, max_h / ih)
    w, h = iw * scale, ih * scale
    st_pdf.append(RLImage(C["O"], width=w, height=h))
    st_pdf.append(Spacer(1, 20))
    st_pdf.append(Paragraph("<b>Conclusion</b>", S["Heading2"]))
    st_pdf.append(Paragraph(
        "The inspected sample was analyzed using automated AI detection models. "
        "Cylindrical dimensions and defect counts were recorded for quality assessment. "
        "This report is intended to support structural evaluation and technical review.",
        S["Normal"]
    ))
    doc.build(st_pdf)
    return C["R"]

# ---------------- FASTAPI ----------------
app = FastAPI(title="Cylinder QC API")
LATEST = {}  # Store latest file paths

@app.post("/predict")
async def predict(file: UploadFile, engineer: str = Form("Engineer"), sample_id: str = Form("Sample-001"), request: Request = None):
    file_path = f"uploaded_{file.filename}"
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    D = C0(file_path)
    df_log = pd.DataFrame([{
        "File": file.filename,
        "Cylinders": int(D["c"]),
        "Voids": int(D["v"]),
        "Surface Pores": int(D["p"]),
        "Honeycombing": int(D["h"]),
        "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }])
    log_path = "inspection_log.csv"
    df_log.to_csv(log_path, mode="a", header=not os.path.exists(log_path), index=False)

    R0(D, engineer=engineer, sample_id=sample_id)

    # Save latest file paths
    LATEST["Report"] = C["R"]
    LATEST["Annotated"] = C["O"]
    LATEST["CSV Log"] = log_path

    base_url = str(request.base_url).rstrip("/") if request else ""

    return JSONResponse(content={
        "File": file.filename,
        "Cylinders": int(D["c"]),
        "Voids": int(D["v"]),
        "Surface Pores": int(D["p"]),
        "Honeycombing": int(D["h"]),
        "Dimensions": [(float(h), float(w)) for h, w in D["dims"]],
        "Report": f"{base_url}/report",
        "Annotated": f"{base_url}/annotated",
        "CSV Log": f"{base_url}/log"
    })

@app.get("/report")
async def download_report():
    path = LATEST.get("Report")
    if path and os.path.exists(path):
        return FileResponse(path, media_type="application/pdf", filename="cylinder_report.pdf")
    return JSONResponse(content={"error": "Report not found"})

@app.get("/annotated")
async def download_annotated():
    path = LATEST.get("Annotated")
    if path and os.path.exists(path):
        return FileResponse(path, media_type="image/jpeg", filename="annotated_result.jpg")
    return JSONResponse(content={"error": "Annotated image not found"})

@app.get("/log")
async def download_log():
    path = LATEST.get("CSV Log")
    if path and os.path.exists(path):
        return FileResponse(path, media_type="text/csv", filename="inspection_log.csv")
    return JSONResponse(content={"error": "CSV log not found"})

