import os 

def tokenization(tokenizer, batched_text, max_length, truncation = True):
    return tokenizer(batched_text['text'], padding = 'max_length', truncation=truncation, 
                        max_length = max_length)

def find_and_create_next_index_dir(directory_path):
    # Get a list of existing indices in the directory
    existing_indices = [name for name in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, name))]

    # Find the maximum index if there are existing indices
    if existing_indices:
        max_index = max([int(index) for index in existing_indices])
        next_index = max_index + 1
    else:
        next_index = 0

    # Create a new directory with the next index
    new_directory_name = str(next_index)
    new_directory_path = os.path.join(directory_path, new_directory_name)
    os.mkdir(new_directory_path)

    return new_directory_path









if __name__ == '__main__':
    new_dir = (find_and_create_next_index_dir('../model_outputs/Clinical-T5-Base_rad_out'))
    print(new_dir)
    os.rmdir(new_dir)

    