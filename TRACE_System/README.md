# TRACE System

TRACE is a full-stack skin analysis platform with:
- React/Vite frontend (`Trace_UI`)
- Flask + MongoDB backend (`TRACE_Backend`)
- Authentication (student/clinician/admin roles)
- Hair-removal analysis workflow with mask + processed output
- Admin control center (users, analytics, analysis records, runtime status)

## Project Structure

```text
TRACE_System/
  TRACE_Backend/   # Flask API + MongoDB + AI processing
  Trace_UI/        # React frontend
```

## Prerequisites

- Node.js 18+
- Python 3.11 (recommended for TensorFlow compatibility)
- MongoDB running locally (default: `mongodb://localhost:27017/trace_db`)

## Backend Setup (Recommended: venv311)

```powershell
cd TRACE_System\TRACE_Backend
py -3.11 -m venv venv311
.\venv311\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

## Frontend Setup

```powershell
cd TRACE_System\Trace_UI
npm install
npm run dev
```

## Environment Variables

Create/update `TRACE_Backend/.env`:

```env
MONGO_URI=mongodb://localhost:27017/trace_db
SECRET_KEY=trace_secret_key_12345
APP_TIMEZONE=Asia/Karachi
```

## Hair Model Notes

- Hair model file path expected:
  - `TRACE_Backend/model/chimaera_v2_final.h5`
- If TensorFlow model is not available, backend can use fallback behavior depending on config.
- Current pipeline returns:
  - mask image
  - mask overlay image
  - processed (hair removed) image

## Admin Panel

Admin routes are protected and require an admin token:
- `/api/admin/users`
- `/api/admin/analytics`
- `/api/admin/analyses`
- `/api/admin/system-status`

Frontend admin path:
- `/admin` (with compatibility redirect from `/dashboard/admin`)

## Git Ignore (Important)

Repository ignores sensitive/heavy content:
- virtual environments (`venv`, `venv311`, `.venv`)
- model binaries (`*.h5`, `*.pth`, `model/`)
- uploads (`uploads/`)
- `.env`
- build outputs (`dist/`, `build/`)
- editor settings (`.vscode/`)

## Run Summary

Use two terminals:
- Terminal 1: backend (`venv311`, `python app.py`)
- Terminal 2: frontend (`npm run dev`)

