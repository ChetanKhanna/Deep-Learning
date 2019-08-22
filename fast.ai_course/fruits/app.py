from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse
from fastai.vision import open_image, load_learner
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio
import os


app = Starlette() # instantiating Starlette app

classes = ['green-apple', 'guava', 'pear']
path = Path(os.getcwd())
model = load_learner(path) # load the pre-trained model

# function to return image(in bytes) from given url
async def get_bytes_from_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

# function to classify image from image passes as bytes
def predict_image_from_bytes(bytes_):
    img = open_image(BytesIO(bytes_))
    predict_class, predict_idx, output = model.predict(img)
    dict_ = dict(zip(classes, output))
    return JSONResponse({
        "predictions": sorted(
            zip(classes, map(float, output)),
            key=lambda p: p[1],
            reverse=True)
        })

# upload-image function
@app.route('/upload', methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes_ = await (data['file'].read())
    return predict_image_from_bytes(bytes_)

# classify-from-url functino
@app.route('/classify-url', methods=["GET"])
async def classify_url(request):
    bytes_ = await get_bytes_from_url(request.querry_params['url'])
    return predict_image_from_bytes(bytes_)

# defining home page for the app
@app.route('/')
def home(request):
    return HTMLResponse("""
        <h3>This app classifies fruits into Green-Apple, Guava or Pear</h3>

        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>

        <form action=/classify-url" method="get">
            Or submit a URL:
            <input type="url" name="url">
            <input type="submit" value="Fetch Image">
        </form>
        """)

# redirectino function
@app.route('/form')
def redirect_to_home(request):
    return RedirectResponse('/')

# starting app
if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app, host='0.0.0.0', port=8000)
