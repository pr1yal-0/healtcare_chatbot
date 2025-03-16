from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd

# Load dataset to get disease labels
df = pd.read_csv("enhanced_disease_dataset.csv")  # Ensure correct path
disease_labels = {d: i for i, d in enumerate(df["Disease"].unique())}
label_to_disease = {v: k for k, v in disease_labels.items()}  # Reverse mapping

# Load model and tokenizer
model_path = "./biobert_model"
model = BertForSequenceClassification.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model.eval()

# Initialize FastAPI app
app = FastAPI()

# Input schema
class SymptomInput(BaseModel):
    symptoms: str

# Prediction function
def predict_disease(symptoms):
    inputs = tokenizer(symptoms, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    predicted_disease = label_to_disease.get(predicted_label, "Unknown Disease")  # Fetch actual disease name
    return predicted_disease

# API Endpoint
@app.post("/predict")
async def predict(input_data: SymptomInput):
    predicted_disease = predict_disease(input_data.symptoms)
    return {"Predicted Disease": predicted_disease}
