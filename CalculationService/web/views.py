import json

from django.http import HttpResponse
from django.shortcuts import render
from django.utils.baseconv import base64
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
import numpy as np
from CalculationService.web import service,result,constant
from CalculationService.web.result import *

def help(request):
    return render(request, "help.html")

@require_http_methods(["POST"])
@csrf_exempt
def compute(request):
    return service.Net.compute(request)
