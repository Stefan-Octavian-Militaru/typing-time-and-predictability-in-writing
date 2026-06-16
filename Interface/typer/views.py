import json
import random
import os
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from typing_predictor import predictor_model

def get_random_prompt():
    prompts_path = os.path.join(settings.BASE_DIR, "prompts.txt")
    with open(prompts_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return random.choice(lines) if lines else "Tell us something interesting about yourself."


def index(request):
    prompt = get_random_prompt()
    return render(request, "typer/index.html", {"prompt": prompt})


@csrf_exempt
def submit(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required"}, status=405)

    try:
        data = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    response_text = data.get("response", "").strip()
    elapsed_ms    = data.get("elapsed_ms", 0)
    prompt        = data.get("prompt", "")
    word_data     = data.get("word_data", [])

    if not response_text:
        return JsonResponse({"error": "Empty response"}, status=400)

    elapsed_s  = elapsed_ms / 1000.0
    words      = response_text.split()
    word_count = len(words)
    wpm        = (word_count / elapsed_s * 60) if elapsed_s > 0 else 0

    sentence_time, word_times = predictor_model.predict(word_data, elapsed_s)

    return JsonResponse({
        "status":      "ok",
        "prompt":      prompt,
        "response":    response_text,
        "elapsed_ms":  elapsed_ms,
        "elapsed_s":   round(elapsed_s, 3),
        "word_count":  word_count,
        "wpm":         round(wpm, 1),
        "word_data":   word_data,
        "prediction":  sentence_time, 
        "word_times":  word_times,      
    })
