def tokenization(tokenizer, batched_text, max_length):
    return tokenizer(batched_text['text'], padding = 'max_length', truncation=True, 
                        max_length = max_length)