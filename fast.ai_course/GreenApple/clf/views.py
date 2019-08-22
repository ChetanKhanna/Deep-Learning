from django.shortcuts import render, redirect
from django.http import HttpResponse
from clf.forms import FruitImageForm
from django.views import generic
from django.conf import settings
from .tasks import predict

import os
# Create your views here.
MEDIA_URL = settings.MEDIA_URL
MEDIA_ROOT = settings.MEDIA_ROOT

IMAGE_PATH = None

class HomeView(generic.TemplateView):
    'Defualt Home view for the web application'
    template_name = 'clf/index.html'
    context = {}
    def post(self, request, *args, **kwargs):
        form = FruitImageForm(request.POST, request.FILES)
        # if succesfully uploaded
        if form.is_valid():
            form.save()
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
        self.context['img_path'] = os.path.join(MEDIA_URL, 'clf', 'Images',
                                                request.session['fruit_image'])
        self.context['abs_img_path'] = os.path.join(MEDIA_ROOT, 'clf', 'Images',
                                                request.session['fruit_image'])
        IMAGE_PATH = str(self.context['abs_img_path'])
        return render(request, self.template_name, self.context)

def PredictResult(request):
    # print(self.context)
    global IMAGE_PATH
    img_path = os.path.join(MEDIA_URL, 'clf', 'Images',
                            os.path.basename(IMAGE_PATH))
    prediction = predict(IMAGE_PATH)
    classes = ['green-apple', 'guava', 'pear']
    dict_ = dict(zip(classes, prediction))
    print(dict_)
    return render(request, 'clf/results.html', {'results': dict_,
                  'img_path': img_path})
