from fastapi import APIRouter, File, HTTPException, Request, UploadFile


router = APIRouter(prefix="", tags=["skin-classifier"])


@router.post("/classify-skin")
async def classify_skin(request: Request, file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    service = getattr(request.app.state, "skin_service", None)
    if service is None:
        raise HTTPException(status_code=500, detail="Skin classifier service is not initialized")

    try:
        return service.predict_image_bytes(image_bytes)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

