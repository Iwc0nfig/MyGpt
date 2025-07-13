import re
from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
import os
from tqdm import tqdm
import numpy as np
from itertools import islice

print('start')

dataset_name = "HuggingFaceFW/fineweb-edu"
subset_name = "sample-10BT" 
output_dir = "./tokenized_data"
texts_per_array = 100000
batch_size = 1000  # The number of texts to process at a time
num_proc = os.cpu_count() #
tokenizer = Tokenizer.from_file("./tokenizer/english_bpe_tokenizer.json")

os.makedirs(output_dir, exist_ok=True)

ascii_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()[]{-}<>@#&$%*+=_/\\|^ "
disallowed_chars_pattern = re.compile(f'[^{re.escape(ascii_chars)}]')

dataset = load_dataset(dataset_name, subset_name , streaming=True)
dataset=dataset['train']

def tokenize_batch(batch):
    # Apply the same character filtering as used during tokenizer training
    filtered_texts = []
    for text in batch['text']:
        # Filter out non-ASCII characters
        filtered_text = disallowed_chars_pattern.sub('', text)
        # Clean up whitespace
        processed_text = filtered_text.strip().replace('\n', ' ')
        filtered_texts.append(processed_text)
    
    tokenized = tokenizer.encode_batch(filtered_texts)
    return {'input_ids': [np.array(t.ids, dtype=np.uint16) for t in tokenized]}


print(f"Starting tokenization with {num_proc} processes...")


tokenized_dataset = dataset.map(
    tokenize_batch,
    batched=True,
    batch_size=batch_size,
    remove_columns=list(dataset.features.keys()) # Remove old columns to save memory
)


tokenized_iterator = iter(tokenized_dataset)
file_counter = 0
is_done = False

while not is_done:
    # A buffer to hold texts for the next numpy file
    buffer = []
    
    # Grab `texts_per_array` items from the iterator
    # Using islice is a memory-efficient way to get a chunk from an iterator
    chunk = list(islice(tokenized_iterator, texts_per_array))
    
    if not chunk:
        # If the chunk is empty, we've processed the whole dataset
        is_done = True
        continue
        
    # Extract the 'input_ids' from each item in the chunk
    for item in chunk:
        buffer.append(item['input_ids'])

    # Save the buffer to a .npy file
    output_filename = os.path.join(output_dir, f"tokenized_chunk_{file_counter}.npy")
    # Using dtype=object because the inner arrays have variable lengths
    np.save(output_filename, np.array(buffer, dtype=object))
    
    print(f"Saved {len(buffer)} tokenized texts to {output_filename}")
    file_counter += 1

print("\nTokenization and saving complete.")
