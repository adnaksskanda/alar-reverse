import yaml
import re
import nltk # NLTK is used for SnowballStemmer and word_tokenize
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize

# --- Initialize Stemmer ---
# You can change "english" to another language supported by SnowballStemmer if needed.
stemmer = SnowballStemmer("english")

def preprocess_and_stem_definitions(data_list):
    """
    Iterates through the loaded YAML data (expected to be a list of dictionaries),
    tokenizes and stems English definition texts found in item['defs'][n]['entry'],
    and adds a list of unique stemmed words as '_stemmed_words_list'
    to each definition dictionary.
    """
    if not isinstance(data_list, list):
        print("Error: Input data is not a list. Please ensure your YAML root is a list.")
        return []

    processed_list = []
    for item_data in data_list:
        if not isinstance(item_data, dict):
            # If an item in the list is not a dictionary, append it as-is or handle as an error.
            # For robustness, we'll append it, but you might want to log a warning.
            processed_list.append(item_data) 
            print(f"Warning: Encountered a non-dictionary item in the input list: {type(item_data)}")
            continue

        new_item_data = item_data.copy() # Process a copy to avoid modifying original during iteration if it were mutable
        
        if 'defs' in new_item_data and isinstance(new_item_data['defs'], list):
            new_defs_list = []
            for defn_item_data in new_item_data['defs']:
                if not isinstance(defn_item_data, dict):
                    new_defs_list.append(defn_item_data) # Append as-is
                    print(f"Warning: Encountered a non-dictionary item within a 'defs' list: {type(defn_item_data)}")
                    continue
                
                new_defn_item = defn_item_data.copy()
                if 'entry' in new_defn_item and isinstance(new_defn_item['entry'], str):
                    definition_text = new_defn_item['entry']
                    
                    # Convert to lowercase before tokenization for consistent stemming
                    definition_text_lower = definition_text.lower()
                    
                    # Tokenize the text into words using NLTK's tokenizer
                    tokens = word_tokenize(definition_text_lower)
                    
                    stemmed_word_set = set() # Use a set to store unique stemmed words
                    for word_token in tokens:
                        # Process if it's a word-like token (alphanumeric or hyphen).
                        # This helps avoid trying to stem pure punctuation.
                        if re.fullmatch(r'[\w-]+', word_token): 
                            stemmed_word_set.add(stemmer.stem(word_token))
                    
                    # Store as a sorted list in the output YAML for readability and consistency
                    new_defn_item['_stemmed_words_list'] = sorted(list(stemmed_word_set))
                else:
                    # If no 'entry' key or its value is not a string, add an empty list
                    new_defn_item['_stemmed_words_list'] = []
                new_defs_list.append(new_defn_item)
            new_item_data['defs'] = new_defs_list
        processed_list.append(new_item_data)
    return processed_list

def main_processing(input_yaml_path="alar.yml", output_yaml_path="alar_stemmed.yml"):
    """
    Main function to load the input YAML, process it by stemming definitions,
    and save the result to the output YAML file.
    """
    print(f"Attempting to load data from: {input_yaml_path}")
    try:
        with open(input_yaml_path, "r", encoding="utf-8") as stream:
            original_data = yaml.safe_load(stream)
    except FileNotFoundError:
        print(f"Error: Input file '{input_yaml_path}' not found. Please make sure it's in the correct directory.")
        return
    except yaml.YAMLError as exc:
        print(f"Error parsing YAML from '{input_yaml_path}': {exc}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading '{input_yaml_path}': {e}")
        return

    if original_data is None:
        print(f"No data was loaded from '{input_yaml_path}'. Output file will not be created.")
        return

    print("Processing definitions (tokenizing and stemming)...")
    processed_data = preprocess_and_stem_definitions(original_data)

    print(f"Attempting to save processed data to: {output_yaml_path}")
    try:
        with open(output_yaml_path, 'w', encoding='utf-8') as outfile:
            # Using SafeDumper for security, allow_unicode for non-ASCII characters,
            # and sort_keys=False to maintain original key order as much as possible in dictionaries.
            yaml.dump(processed_data, outfile, allow_unicode=True, sort_keys=False, Dumper=yaml.SafeDumper)
        print(f"Processing complete. Output saved to '{output_yaml_path}'")
    except Exception as e:
        print(f"An error occurred while saving to '{output_yaml_path}': {e}")

if __name__ == "__main__":
    # Define your input and output file names here
    # Ensure 'alar.yml' (or your input file) is in the same directory as this script,
    # or provide the full path.
    INPUT_FILE = "alar.yml" 
    OUTPUT_FILE = "alar_stemmed.yml"
    
    # Basic check for NLTK's 'punkt' resource needed by word_tokenize
    try:
        word_tokenize("test sentence for tokenizer") # A simple call to see if punkt is available
    except LookupError:
        print("NLTK 'punkt' resource not found. This is required for word tokenization.")
        print("Please download it by running the following in a Python interpreter:")
        print("  import nltk")
        print("  nltk.download('punkt')")
        print("-" * 30)
        # Decide if you want to exit or let main_processing try and potentially fail
        print("Script will attempt to continue, but tokenization might fail if 'punkt' is missing.")
    
    main_processing(INPUT_FILE, OUTPUT_FILE)
