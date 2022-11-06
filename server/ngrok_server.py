import os
import sys
from face_animation import FaceAnimator
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import nest_asyncio
from pyngrok import ngrok
import uvicorn
import uuid

app = FastAPI()
face_animator = FaceAnimator()

@app.get('/index')
async def index():
  # save received file 
  src_path = "../../input.jpg" # todo
  # generate temporary destination path
  dst_path = str(uuid.uuid4()) + ".png"
  try:
    animation_successful = face_animator.process(src_path, dst_path)
  except:
    raise HTTPException(status_code=500, detail=sys.exc_info())
  if not animation_successful:
    raise HTTPException(status_code=204, detail="no face suitable for animation found")
  response = FileResponse(dst_path)
  #os.remove(dst_path)
  return response

def main():
  ngrok_tunnel = ngrok.connect(8000)
  print('Public URL:', ngrok_tunnel.public_url)
  nest_asyncio.apply()
  uvicorn.run(app, port=8000)

if __name__ == "__main__":
  main()