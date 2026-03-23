# TRACE System – Skin Cancer Detection Platform

## System Overview

TRACE is a full-stack AI-assisted dermatology workflow:

- The **Frontend** provides authentication, upload, analysis, and result visualization.
- The **Backend** exposes API endpoints for preprocessing, skin cancer classification, and lesion localization.
- Users upload dermoscopic images and receive model-driven outputs in a structured medical-style interface.

## Technical Stack

- **Frontend**: React + Vite + Tailwind CSS
- **Backend**: Flask + Python
- **AI Models**:
  - TensorFlow/Keras model for hair-removal preprocessing
  - PyTorch ConvNeXt for 4-class skin cancer classification
  - YOLO for lesion localization

## End-to-End Data Flow

1. User uploads an image in the frontend (`Trace_UI`).
2. Frontend sends a `POST` request with `multipart/form-data` to backend `/predict`.
3. Backend validates and processes the image:
   - Hair-removal preprocessing
   - ConvNeXt classification
   - YOLO localization
4. Backend returns JSON containing:
   - classification details (class, confidence, severity, diagnosis fields)
   - localization details (bbox coordinates, confidence, labels)
   - preprocessing artifacts (mask/processed image references)
5. Frontend renders:
   - medical-report style diagnosis card
   - probability/severity/recommendation
   - lesion bounding box overlay on processed image

## Repository Layout (dev2)

```text
TRACE-Skin-Cancer-Detection/
├── FRONTEND_REPORT.md
├── README.md
├── .gitignore
├── Trace_UI/
│   ├── src/components
│   ├── src/pages
│   ├── src/services
│   └── src/assets
└── Backend/
    ├── app.py
    ├── requirements.txt
    ├── services/
    └── endpoints/
```

## Setup Instructions (Local)

### 1) Frontend

```bash
cd Trace_UI
npm install
npm run dev
```

### 2) Backend

```bash
cd Backend
pip install -r requirements.txt
python app.py
```

### 3) Configuration

- Set required environment variables (DB URI, JWT secret, email settings, model paths).
- Keep local secrets in `.env` (ignored by git).

## Viva Talking Points

- Explain why service-layer extraction in frontend (`src/services`) improves maintainability.
- Show how frontend-to-backend API contracts drive UI rendering.
- Discuss model pipeline ordering (preprocess -> classify -> localize).
- Highlight repository hygiene and branch discipline (`dev2` only, selective staging, no accidental backend/debug artifacts).

## Conclusion

This branch is structured for collaboration, demonstration, and pull-request readiness with cleaner frontend organization and clear full-stack documentation.
