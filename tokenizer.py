from datasets import load_dataset
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import Lowercase, NFD, StripAccents, Sequence
import os
import re

dataset_name = "HuggingFaceFW/fineweb-edu"
subset_name = "sample-10BT" 
docs_for_tokenizer = 100_000   
file_sample = "tokenizer_corpus.txt"
vocab_size = 16_000
ascii_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?;:'\"()[]{-}<>@#&$%*+=_/\\|^ "
ascii_tokens = list(ascii_chars)

disallowed_chars_pattern = re.compile(f'[^{re.escape(ascii_chars)}]')
dataset = load_dataset(dataset_name, subset_name)
train = dataset['train']

print("Sampling data from FineWeb-Edu...")
sampled_lines = []
for item in train:
    # Filter the text to keep only the allowed characters
    text = item['text']
    filtered_text = disallowed_chars_pattern.sub('', text)
    # Clean up whitespace and ensure the line has meaningful content
    processed_text = filtered_text.strip().replace('\n', ' ')
    if processed_text and len(processed_text) > 30:
        sampled_lines.append(processed_text)

    # Stop once we have enough documents
    if len(sampled_lines) >= docs_for_tokenizer:
        break



with open(file_sample,'w', encoding='utf-8') as f:
    for line in sampled_lines:
        f.write(line + '\n')


print("✅ Saved tokenizer_corpus.txt with", len(sampled_lines), "lines.")


# Step 3: Train tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

trainer = trainers.BpeTrainer(
    vocab_size=16000,
    initial_alphabet=ascii_tokens, 
    special_tokens=["<pad>", "<unk>", "<eos>"]
)

tokenizer.train(["tokenizer_corpus.txt"], trainer)
tokenizer.decoder = decoders.BPEDecoder()


tokenizer.post_processor = processors.TemplateProcessing(
    single="$A <eos>",
    pair="$A <eos> $B:1 <eos>:1",
    special_tokens=[("<eos>", 2)]
)


os.makedirs("./tokenizer", exist_ok=True)
tokenizer.save("./tokenizer/english_bpe_tokenizer.json")


