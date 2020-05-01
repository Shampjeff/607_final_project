from django.shortcuts import render
from subprocess import run, PIPE

import sys
import pandas as pd


def empty_click(request):
	return render(request, 'home.html')

def output(request):
	data=external(request)
	return render(request, 'home.html', {"data":data})

def external(request):
	input_text=request.POST.get("input_text")
	input_text_2=request.POST.get("input_text_2")
	input_text_3=request.POST.get("input_text_3")
	input_text_4=request.POST.get("input_text_4")
	input_text_5=request.POST.get("input_text_5")

	out = run([sys.executable,
	 #"//Users//jeffshamp//Documents//607_final_project//prediction_engine.py",
     "..//..//prediction_engine.py",
	  input_text, input_text_2, input_text_3, input_text_4, input_text_5,
	  "source_a", "source_b", "source_c","source_d","source_e"],
	 shell=False, 
	 stdout=PIPE)
	return render(request, "home.html", {"results_text":out.stdout})