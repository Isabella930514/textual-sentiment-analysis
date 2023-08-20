import os
import threading
import pickle
import torch

from tqdm import tqdm
from textblob import TextBlob
from transformers import BertTokenizer, BertModel

DIM = 100
num_threads = 3
max_words = 100
MSG_SAVED_DATA = True

local_model = BertModel.from_pretrained('bert-large-uncased')
local_tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

class Review:
    def __init__(self, text, local_model, local_tokenizer):
        self.text = text
        self.tensor = None
        self.normalized_sentiment = None

        self.generate_tensor(local_model, local_tokenizer)
        self.generate_sentiment()

    def generate_sentiment(self):
        blob = TextBlob(self.text)
        self.normalized_sentiment = (blob.sentiment.polarity + 1) / 2

    def generate_tensor(self, local_model, local_tokenizer):
        total_str = self.text
        sentence = str(total_str.split(' ')[:max_words]) if len(total_str.split(' ')) > max_words else total_str
        tokens = local_tokenizer.encode(sentence, add_special_tokens=True)
        input_ids = torch.tensor(tokens).unsqueeze(0)
        with torch.no_grad():
            outputs = local_model(input_ids)
        embedding = outputs[0].squeeze(0)[0]
        self.tensor = embedding[:DIM]

def process_lines(lines, result):
    for line in tqdm(lines):
        try:
            data = line.split("\t")
            item_id = data[0]
            text = data[1]
            review = Review(text, local_model, local_tokenizer)
            result[item_id] = review
        except:
            pass

# Load the function from a file
def load_function(filename):
    with open(filename, 'rb') as f:
        func = pickle.load(f)
    return func

# Save the function to a file
def save_function(func, filename):
    with open(filename, 'wb') as f:
        pickle.dump(func, f)

def textual_sentiment_analysis(filename):
    if os.path.exists(filename + ".pkl") and MSG_SAVED_DATA:
        return load_function(filename + ".pkl")

    with open(filename, encoding='utf-8') as file:
        lines = file.readlines()

    result = {}
    chunk_size = len(lines) // num_threads

    # Create and start worker threads
    threads = []
    for i in range(num_threads):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_threads - 1 else len(lines)
        thread = threading.Thread(target=process_lines, args=(lines[start:end], result))
        thread.start()
        threads.append(thread)

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    save_function(result, filename + ".pkl")

    return result

def main():

    textual_sentiment_analysis("./data.tsv")

if __name__ == '__main__':
    main()
