from django.shortcuts import render, redirect
from django.http import HttpResponse
from clf.forms import FruitImageForm
from django.views import generic
from django.conf import settings
from .tasks import predict

import os


# getting paths for Media storage (abs and relative)
MEDIA_URL = settings.MEDIA_URL
MEDIA_ROOT = settings.MEDIA_ROOT

IMAGE_PATH = None # gloabal variable for uploaded image path

class HomeView(generic.TemplateView):
    'Defualt Home view for the web application'
    
    template_name = 'clf/index.html'
    context = {}

    def post(self, request, *args, **kwargs):
        # clearning media if anything present before
        img_folder = os.path.join(MEDIA_ROOT, 'clf', 'Images')
        for file in os.listdir(img_folder):
            img_file = os.path.join(img_folder, file)
            try:
                os.remove(img_file)
            except Exception as e:
                print(e)
        # getting form
        form = FruitImageForm(request.POST, request.FILES)
        # if succesfully uploaded
        if form.is_valid():
            form.save()
            # save file name for later refernce
            request.session['fruit_image'] = request.FILES['fruit_image'].name
            return redirect('/analysis')
        self.context['form'] = form
        return render(request, self.template_name, self.context)

    def get(self, request, *args, **kwargs):
        return render(request, self.template_name, self.context)
    

class RunAnalysisView(generic.TemplateView):
    'Displays uploaded photo and option to run prediction'
    template_name = 'clf/analysis.html'
    context = {}

    def get(self, request, *args, **kwargs):
        global IMAGE_PATH
        # absolute path for fastai model and relative path for django template
        self.context['img_path'] = os.path.join(MEDIA_URL, 'clf', 'Images',
                                                request.session['fruit_image'])
        self.context['abs_img_path'] = os.path.join(MEDIA_ROOT, 'clf', 'Images',
                                                request.session['fruit_image'])
        IMAGE_PATH = str(self.context['abs_img_path']) # storing globally
        return render(request, self.template_name, self.context)

def PredictResult(request):
    global IMAGE_PATH # abs path for  model
    img_path = os.path.join(MEDIA_URL, 'clf', 'Images',
                            os.path.basename(IMAGE_PATH)) # for template
    prediction = predict(IMAGE_PATH) # getting the prediction from model
    classes = ['green-apple', 'guava', 'pear']
    dict_ = dict(zip(classes, prediction)) # zipping in a dictinoary
    return render(request, 'clf/results.html', {'results': dict_,
                  'img_path': img_path})
