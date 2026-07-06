from fastapi import APIRouter
from fastapi import UploadFile
from fastapi.responses import StreamingResponse
from fastapi import HTTPException

from app.inference import BackgroundService


router=APIRouter()

service=BackgroundService()


@router.get("/health")
async def health():

    return {
        "status":"ok"
    }


@router.post("/remove-bg")
async def remove_bg(
    file:UploadFile
):

    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Invalid image"
        )

    image=await file.read()

    result=service.remove_background(
        image
    )

    return StreamingResponse(
        result,
        media_type="image/png"
    )