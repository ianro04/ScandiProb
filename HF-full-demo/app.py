import os
import re
import torch
import pypdf
import gradio as gr
from ftfy import TextFixerConfig, fix_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from docx import Document

HF_TOKEN = os.environ.get("HF_TOKEN")
repo_id = "ianro04/ScandiProb"

LANG_LABELS = ["Norwegian", "Swedish", "Danish", "Non-Scandinavian"]
FTFY_CONFIG=TextFixerConfig(uncurl_quotes=False)

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(repo_id, token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained(repo_id, token=HF_TOKEN)
model.eval()

def read_file(file_path): # Alt input method for Space
    if file_path is None:
        return ""
    if file_path.lower().endswith((".txt", ".md", ".rtf")):
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    elif file_path.lower().endswith(".docx"):
        doc = Document(file_path)
        return "\n".join([p.text for p in doc.paragraphs])
    elif file_path.lower().endswith(".pdf"):
        reader = pypdf.PdfReader(file_path)
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    return ""

def nonscandi_penalty(text): # Copy-pasting everything that isn't the raw model here
    text = text.strip()
    if len(text) < 2:
        return 1.0

    scandi_keyboard = r"[a-zA-ZæøåÆØÅäöÄÖéÉ0-9 !@#$%^&*()\-_=+\[\]{};':\",.<>?/`~\\|]"
    scandi_keyboard_alpha_only = r"[a-zA-ZæøåÆØÅäöÄÖéÉ ]"
    scandi_key_matches = re.findall(scandi_keyboard, text)
    scandi_alpha_matches = re.findall(scandi_keyboard_alpha_only, text)
    
    if len(scandi_alpha_matches) < (len(text) * 0.5):
        nonscandi_percent = 1.0
    else:
        nonscandi_percent = (1 - (len(scandi_key_matches) / len(text)))
    return nonscandi_percent

def da_no_cross_skew(text):
    text = text.strip().lower()
    if not text:
        return [0.0, 0.0]
    
    da_skew, no_skew = 0.0, 0.0
    
    da_no_regex = {
        r"æ[bgltv]": "DA", 
        r"[eø]j" : "DA", 
        r"\b\w+hed(?:en|et)?\b" : "DA",
        r"\b\w*([bdfgklnprst])\1\b" : "NO", 
        r"(?:g|k|sk)j[eæø]" : "NO"
    }
    
    words = text.split()
    if not words:
        return [0.0, 0.0]
        
    skew_amount = 1.0 / len(text)
    
    for rule, lang in da_no_regex.items():
        rule_matches = len(re.findall(rule, text))
        skew_inc = rule_matches * skew_amount * (2 if len(words) <= 6 else 1)
        if lang == "NO":
            no_skew += skew_inc
            da_skew -= skew_inc
        elif lang == "DA":
            da_skew += skew_inc
            no_skew -= skew_inc
            
    return [no_skew, da_skew]

def ScandiProb(text):
    text = text.strip()
    if not text:
        return "None", {label: 0.0 for label in LANG_LABELS}
        
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)
        
    raw_probs = torch.sigmoid(outputs.logits)[0]
    
    nonscandi_ratio = nonscandi_penalty(text)
    no_skew, da_skew = da_no_cross_skew(text)
    
    final_probs = {}
    
    for i, label in enumerate(LANG_LABELS):
        prob = raw_probs[i].item()
        
        if label in ["Norwegian", "Swedish", "Danish"]:
            adjusted = prob * (1.0 - nonscandi_ratio)
        else:
            adjusted = prob + ((1.0 - prob) * nonscandi_ratio)

        if label == "Norwegian":
            adjusted = adjusted * (1.0 + no_skew)
            adjusted = adjusted * (1.0 - da_skew)
        elif label == "Danish":
            adjusted = adjusted * (1.0 + da_skew)
            adjusted = adjusted * (1.0 - no_skew)

        adjusted = min(1.0, max(0.0, adjusted))
        final_probs[label] = float(adjusted)

    top_labels = [label for label, prob in final_probs.items() if prob > 0.5]
    
    top_labels_str = ", ".join(top_labels) if top_labels else "Indefinitive"
        
    return top_labels_str, final_probs

def classify(text, file):
    if file is not None:
        text = fix_text(read_file(file), config=FTFY_CONFIG)
    return ScandiProb(text)

with gr.Blocks() as demo:
    gr.Markdown("# ScandiProb: Hybrid Language ID Classifier")
    gr.Markdown("### By Ian Rodriguez")
    gr.Markdown("Enter text or upload a file to output independent probabilities that it is written in **Norwegian**, **Swedish**, **Danish**, or **None of the Above / Non-Scandinavian**. Only the first 512 tokens of input will be used.")
    gr.Markdown("This model utilizes a fine-tuned *ScandiBERT*, trained on limited amounts of *OPUS-100*, and combined with regex-enforced heuristics. Achieves ~93% macro-F1 score on OPUS-100 test set and ~84% macro-F1 score against the comprehensive SLIDE eval set, with a fraction of the training data used in SLIDE.")
    
    with gr.Row():
        with gr.Column():
            with gr.Tab("Text Input"):
                input_text = gr.Textbox(lines=5, placeholder="Enter text...", label="Input Text")
            with gr.Tab("File Upload"):
                input_file = gr.File(file_types=[".txt", ".docx", ".pdf", ".md", ".rtf"])
            submit_btn = gr.Button("Classify")
        
        with gr.Column():
            top_prediction = gr.Textbox(label="Probable Languages (>50%)", interactive=False)
            output_labels = gr.Label(num_top_classes=4, label="All Probabilities")
            
    submit_btn.click(fn=classify, inputs=[input_text, input_file], outputs=[top_prediction, output_labels])

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Ocean())
