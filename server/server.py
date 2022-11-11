import os
import sys
import traceback
from face_animation import FaceAnimator
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
face_animator = FaceAnimator()
port = sys.argv(sys.argv.index("--port") + 1) if "--port" in sys.argv else 8000

@app.post('/animate')
async def index(data: Data):
    try:
        url = data.img_url
        # save received file to temporary path
        img = Image.open(urlopen(url))
        src_path = str(uuid.uuid4()) + "." + img.format
        img.save(src_path)
        # generate temporary destination path
        dst_path = str(uuid.uuid4()) + ".png"
        face_found = face_animator.process(src_path, dst_path)
        if(face_found):
            response = FileResponse(dst_path)
            return response
    except:
        raise HTTPException(status_code=500, detail=traceback.format_exc())
    if not face_found:
        raise HTTPException(status_code=204, detail="no face suitable for animation found")

def main():
    nest_asyncio.apply()
    uvicorn.run(app, port=port)

if __name__ == "__main__":
    main()