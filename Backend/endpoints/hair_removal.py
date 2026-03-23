from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from model_logic import encode_processed_jpeg, process_image_bytes

router = APIRouter(prefix="", tags=["hair-removal"])


@router.post("/remove-hair")
async def remove_hair(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Upload a valid image file")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image file")

    try:
        processed_bgr = process_image_bytes(image_bytes)
        output_bytes = encode_processed_jpeg(processed_bgr)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Return exact processed image bytes for frontend consumption.
    return StreamingResponse(
        iter([output_bytes]),
        media_type="image/jpeg",
        headers={"Content-Disposition": "inline; filename=processed.jpg"},
    )

