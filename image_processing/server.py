# In the wink of an eye
# an extension for Google Chrome that makes images of 
# Czech president Miloš Zeman more realistic
#
# Copyright (C) 2022 Tomáš Nekut
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You received a copy of the GNU General Public License
# along with this program or you can find it at <https://www.gnu.org/licenses/>.

import sys
import traceback
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
from face_utils.animation import FaceAnimator

class Data(BaseModel):
    img_url: str

def get_argument(name, default=None):
    try: return sys.argv[sys.argv.index(name) + 1]
    except:
        if default==None:
            print("You have to specify '" + name + "' parameter."), exit(0)
        return default

port = int(get_argument("--port"))
face_example = get_argument("--face_example", default="assets/zeman.jpg")
motion_vectors = get_argument("--motion_vectors", default="assets/wink.npy")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
face_animator = FaceAnimator(face_example=face_example, motion_vectors=motion_vectors)

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