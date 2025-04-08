from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from collections import Counter

class HateSpeechDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()} | {
            "labels": torch.tensor(self.labels[idx])
        }

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=1)
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions)
    }

def train():
    df = pd.read_csv('data/dataset.csv')

    label_mapping = {'hate': 1, 'neutral': 0}
    df['label'] = df['label'].map(label_mapping)

    train_texts, val_texts, train_labels, val_labels = train_test_split(
        df['text'].tolist(), df['label'].tolist(), test_size=0.2, random_state=42)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-multilingual-uncased')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_encodings = tokenizer(train_texts, truncation=True, padding=True, return_tensors="pt")
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, return_tensors="pt")

    train_dataset = HateSpeechDataset(train_encodings, train_labels)
    val_dataset = HateSpeechDataset(val_encodings, val_labels)

    training_args = TrainingArguments(
        output_dir='./models/pretrained',
        num_train_epochs=20,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        eval_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    tokenizer.save_pretrained('./models/pretrained')
    model.save_pretrained('./models/pretrained')

    # all_words = []

    # combined_texts = train_texts + val_texts

    # for text in combined_texts:
    #     words = text.split()
    #     all_words.extend(words)

    # word_counts = Counter(all_words)

    # unique_words = list(word_counts.keys())

    # special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
    # vocab_list = special_tokens + unique_words

    # with open('./models/checkpoint-360/vocab.txt', 'w', encoding='utf-8') as f:
    #     for word in vocab_list:
    #         f.write(word + '\n')

if __name__ == "__main__":
    train()
    