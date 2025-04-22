from flask import Flask, request, render_template
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import numpy as np


app = Flask(__name__)


MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def classify_text(text):
    """Classifies sentiment of input text."""
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        output = model(**tokens)
    scores = softmax(output.logits.numpy()[0])
    labels = ["Negative", "Neutral", "Positive"]
    max_score_index = np.argmax(scores)
    return labels[max_score_index]


@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        text = request.form['text']
        result = classify_text(text)
    return render_template('index.html', result=result)
if __name__ == '__main__':
    app.run(debug=True)
