from image_alteration import alter_image
from fastapi import FastAPI
from fastapi.responses import FileResponse
import nest_asyncio
from pyngrok import ngrok
import uvicorn

app = FastAPI()

@app.get('/')
async def home():
  alter_image("../../input.jpg", "out.gif")
  return FileResponse("out.gif")

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)