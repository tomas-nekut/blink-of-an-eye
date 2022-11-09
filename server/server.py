import os
import sys
import traceback
from face_animation import FaceAnimator
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import nest_asyncio
import uvicorn
import uuid
from PIL import Image
from urllib.request import urlopen
from PIL import Image

class Data(BaseModel):
    img_url: str

app = FastAPI()
face_animator = FaceAnimator()

@app.post('/')
async def index(data: Data):
    try:
        url = data.img_url
        # save received file to temporary path
        print(url)
        img = Image.open(urlopen(url))
        src_path = str(uuid.uuid4()) + "." + img.format
        img.save(src_path)
        # generate temporary destination path
        dst_path = str(uuid.uuid4()) + ".png"
        face_found = face_animator.process(src_path, dst_path)
        if(face_found):
            response = FileResponse(dst_path)
            os.remove(dst_path)
            return response
    except:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    if not face_found:
        raise HTTPException(status_code=204, detail="no face suitable for animation found")

def main():
    nest_asyncio.apply()
    uvicorn.run(app, port=8000)

if __name__ == "__main__":
    main()