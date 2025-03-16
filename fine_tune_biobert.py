import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
import pandas as pd

# Load dataset
df = pd.read_csv("c:/Users/lpriy/Desktop/internship/chatbot/enhanced_disease_dataset.csv")

# Combine symptoms into a single text field
df["Symptoms"] = df.iloc[:, 1:].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Define tokenizer
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")

# Encode labels
disease_labels = {d: i for i, d in enumerate(df["Disease"].unique())}
df["Label"] = df["Disease"].map(disease_labels)

# Train-test split
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["Symptoms"].tolist(), df["Label"].tolist(), test_size=0.2, random_state=42)

# Tokenization
def tokenize_function(texts):
    return tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")

train_encodings = tokenize_function(train_texts)
val_encodings = tokenize_function(val_texts)

# Define PyTorch Dataset
class MedicalDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

train_dataset = MedicalDataset(train_encodings, train_labels)
val_dataset = MedicalDataset(val_encodings, val_labels)

# Load pre-trained BERT model
model = BertForSequenceClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.1", num_labels=len(disease_labels))

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch"
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Train model
trainer.train()

# Save model
model.save_pretrained("./biobert_model")
tokenizer.save_pretrained("./biobert_model")

print("Training completed! Model saved in './biobert_model'")

# Function to predict disease from input symptoms
def predict_disease(symptoms):
    model.eval()
    inputs = tokenizer(symptoms, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_label = torch.argmax(outputs.logits, dim=1).item()
    predicted_disease = list(disease_labels.keys())[list(disease_labels.values()).index(predicted_label)]
    return predicted_disease

# Continuous user input for testing
if __name__ == "__main__":
    while True:
        user_input = input("Enter symptoms (comma-separated) or type 'exit' to quit: ")
        if user_input.lower() == 'exit':
            print("Exiting test mode...")
            break
        predicted = predict_disease(user_input)
        print(f"Predicted Disease: {predicted}")
