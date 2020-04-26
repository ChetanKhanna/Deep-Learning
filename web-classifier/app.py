from starlette.applications import Starlette
from starlette.responses import RedirectResponse
from starlette.templating import Jinja2Templates
from fastai.vision import open_image, load_learner
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
#import asyncio
import os

app = Starlette() # instantiating Starlette app

classes = ['green-apple', 'guava', 'pear']
path = Path(os.getcwd())
model = load_learner(path) # load the pre-trained model

templates = Jinja2Templates(directory='data/Templates')

# function to return image(in bytes) from given url
async def get_bytes_from_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

# function to classify image from image passes as bytes
def predict_image_from_bytes(request, bytes_):
    img = open_image(BytesIO(bytes_))
    _, _, output = model.predict(img)
    prediction = sorted(zip(classes, map(float, output)),
                        key=lambda p: p[1], reverse=True)
    return templates.TemplateResponse('result.html',
                                     {'request': request,
                                      'prediction': prediction})

# upload-image function
@app.route('/upload', methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes_ = await (data['file'].read())
    return predict_image_from_bytes(request, bytes_)

# classify-from-url functino
@app.route('/classify-url', methods=["GET"])
async def classify_url(request):
    bytes_ = await get_bytes_from_url(request.querry_params['url'])
    return predict_image_from_bytes(request, bytes_)

# defining home page for the app
@app.route('/')
def home(request):
    return templates.TemplateResponse('index.html', {'request': request})

# redirectino function
@app.route('/form')
def redirect_to_home(request):
    return RedirectResponse('/')

# starting app
if __name__ == '__main__':
    if 'serve' in sys.argv:
        uvicorn.run(app, host='0.0.0.0', port=8000)
