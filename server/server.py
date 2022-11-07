
import sys
from flask import Flask, send_file, make_response, request
from flask_lt import run_with_lt
from flask_cors import CORS
from urllib.request import urlopen
from PIL import Image
import uuid
from face_animation import FaceAnimator
import traceback

app = Flask(__name__)
CORS(app, origins='*')
face_animator = FaceAnimator()

if "-localtunel" in sys.argv:
    run_with_lt(app, subdomain="in-a-blink-of-an-eye")

@app.route("/", methods=['POST'])
def index(): 
    url = request.json['img_url']
    # save received file to temporary path
    img = Image.open(urlopen(url)).convert('RGB')
    src_path = str(uuid.uuid4()) + "." + img.format
    img.save(src_path)
    # generate temporary destination path
    dst_path = str(uuid.uuid4()) + ".png"
    try:
        animation_successful = face_animator.process(src_path, dst_path)
    except:
        return traceback.format_exc(), 500
    if not animation_successful:
        return "no face suitable for animation found", 204
    
    response = make_response(send_file(dst_path))
    #response.headers['Access-Control-Allow-Origin'] = '*'
    #os.remove(dst_path)
    return response

if __name__ == '__main__':
    app.run()