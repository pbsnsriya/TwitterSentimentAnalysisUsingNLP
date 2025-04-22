import tkinter as tk
from tkinter import filedialog, scrolledtext
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load Model and Tokenizer
MODEL_NAME = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

def preprocess_text(text):
    """Preprocess input text."""
    new_text = []
    for t in text.split(" "):
        t = '@user' if t.startswith('@') and len(t) > 1 else t
        t = 'http' if t.startswith('http') else t
        new_text.append(t)
    return " ".join(new_text)

def classify_text():
    """Classify the input text and display output."""
    text = text_input.get("1.0", tk.END).strip()
    if not text:
        output_display.insert(tk.END, "Please enter some text!\n")
        return
    
    output_display.delete("1.0", tk.END)  # Clear previous output
    processed_text = preprocess_text(text)
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=1).item()
    
    sentiment_labels = ["Negative", "Neutral", "Positive"]
    sentiment = sentiment_labels[prediction]
    output_display.insert(tk.END, f"Sentiment: {sentiment}\n")

def open_file():
    """Open a file and classify its contents."""
    file_path = filedialog.askopenfilename()
    if file_path:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        text_input.delete("1.0", tk.END)
        text_input.insert(tk.END, text)

# Create GUI Window
root = tk.Tk()
root.title("Text Sentiment Analyzer")
root.geometry("600x400")

# Input Text Box
text_input = scrolledtext.ScrolledText(root, height=5, width=60)
text_input.pack(pady=10)

# Buttons
btn_classify = tk.Button(root, text="Classify Text", command=classify_text)
btn_classify.pack()

btn_open = tk.Button(root, text="Open File", command=open_file)
btn_open.pack()

# Output Display
output_display = scrolledtext.ScrolledText(root, height=10, width=60)
output_display.pack(pady=10)

# Run Application
root.mainloop()