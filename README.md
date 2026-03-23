# TRACE System – Frontend (Skin Cancer Detection Platform)

This branch contains a clean integration of the TRACE project with a professional structure for demo, evaluation, and pull-request review.

## Project Structure

```text
TRACE-Skin-Cancer-Detection/
├── FRONTEND_REPORT.md
├── README.md
├── .gitignore
├── Trace_UI/
└── Backend/
```

## Technology Stack

- Frontend: React, Vite, Tailwind CSS
- Backend: Flask + Python
- ML Inference: TensorFlow/Keras (hair removal), PyTorch ConvNeXt (classification), YOLO (localization)

## Run Locally

### Frontend

```bash
cd Trace_UI
npm install
npm run dev
```

### Backend

```bash
cd Backend
pip install -r requirements.txt
python app.py
```

## Notes

- Configure environment variables before running production/demo builds.
- Keep large trained model files in `Backend/model/` (or configured model paths) and avoid committing local secrets.
