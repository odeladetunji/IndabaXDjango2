import os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files import File
from .models import ImageHandler
import io
from django.core.files.storage import FileSystemStorage
import shutil


def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

@csrf_exempt
def logOutfile(request):
    print(request.FILES)
    print(request.FILES['firstImage'])
    fileObject = request.FILES['firstImage']
    fs = FileSystemStorage()
    filename = fs.save(fileObject.name, fileObject)
    uploaded_file_url = fs.url(filename)
    shutil.move(fileObject.name, "./static/" + fileObject.name)
    return HttpResponse(uploaded_file_url)    