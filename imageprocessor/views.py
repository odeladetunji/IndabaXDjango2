import os
from django.http import HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files import File
from .models import ImageHandler
from .src import predict_app
import io
from django.core.files.storage import FileSystemStorage
import shutil
import json
import osgeo

def index(request):
    return HttpResponse("Hello, world. You're at the polls index.")

@csrf_exempt
def logOutfile(request):
    # print(request.GET['firstImage'])
    fileObject = request.FILES['firstImage']
    fs = FileSystemStorage()
    filename = fs.save(fileObject.name, fileObject)
    uploaded_file_url = fs.url(filename)
    shutil.move(fileObject.name, "./static/" + fileObject.name)

    img_path = "./static/" + fileObject.name
    # convertToPNG(img_path)
    model_path= "./imageprocessor/checkpoints/final_model.h5"
    prediction_result = predict_app.main(img_path, model_path)
    # result = json.dumps(prediction_result)
    # print(result)
    return HttpResponse(json.dumps(eval(str(prediction_result))))

    # return HttpResponse(prediction_result)
    # return HttpResponse(uploaded_file_url)    


# def convertToPNG(path_file):  
#     # file_path="D:/work/python/Tif_to_png/a_image.tif"
#     print(path_file)
#     file_path = path_file
#     ds=osgeo.gdal.Open(file_path)
#     driver=osgeo.gdal.GetDriverByName('PNG')
#     dst_ds = driver.CreateCopy(file_path, ds)
#     dst_ds = None
#     src_ds = None
