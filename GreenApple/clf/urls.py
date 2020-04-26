from django.urls import include, path
from django.conf.urls import url
from clf import views
from django.conf import settings
from django.conf.urls.static import static

app_name = 'clf'

urlpatterns = [
	path('', views.HomeView.as_view()),
	path('analysis', views.RunAnalysisView.as_view()),
	path('predict', views.PredictResult)
]
if settings.DEBUG:
	urlpatterns += static(settings.MEDIA_URL,
						  document_root=settings.MEDIA_ROOT)