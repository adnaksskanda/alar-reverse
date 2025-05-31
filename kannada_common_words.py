import sqlite3
import os
import re
import json # For reading JSON files
from collections import Counter
import argparse

# Attempt to import Indic NLP Library for better tokenization
try:
    from indicnlp.tokenize import indic_tokenize
    INDIC_NLP_AVAILABLE = True
    print("Using 'indic-nlp-library' for Kannada tokenization.")
except ImportError:
    INDIC_NLP_AVAILABLE = False
    print("Warning: 'indic-nlp-library' not found. Using basic regex-based tokenizer for Kannada. "
          "For better accuracy, please install with: pip install indic-nlp-library")

# Optional: tqdm for progress bar
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

def clean_kannada_text(text):
    """
    Basic cleaning for Kannada text.
    - Removes common English/Latin characters and numbers if they are isolated words.
    - Keeps Kannada characters.
    """
    # Remove English characters and numbers - adjust regex if needed
    text = re.sub(r'[a-zA-Z0-9]+', '', text) 
    # Retain Kannada characters and spaces. Remove other symbols that are not typical word constituents.
    # \u0C80-\u0CFF is the Kannada Unicode range.
    text = re.sub(r'[^\u0C80-\u0CFF\s]', '', text)
    # text = text.lower() # Lowercasing is generally not applicable or standard for Kannada frequency.
    return text

def tokenize_kannada(text, use_indic_nlp=True):
    """
    Tokenizes Kannada text.
    Uses indic_tokenize if available and use_indic_nlp is True, otherwise basic regex split.
    """
    if use_indic_nlp and INDIC_NLP_AVAILABLE:
        # indic_tokenize.trivial_tokenize splits based on space and some punctuation.
        # For more sophisticated tokenization, you might explore other functions in indic_nlp_library
        # or specific Kannada tokenizers if available.
        return indic_tokenize.trivial_tokenize(text, lang='kn')
    else:
        # Basic regex to find sequences of Kannada characters as words
        tokens = re.findall(r'[\u0C80-\u0CFF]+', text)
        return tokens

def extract_text_from_json_data(data, text_keys):
    """
    Recursively extracts text strings from specified keys in JSON data.
    Handles lists of objects, a single object at the root, or nested structures
    to some extent by processing list items recursively.
    """
    all_text_snippets = []
    if isinstance(data, dict):
        for key in text_keys:
            if key in data:
                value = data[key]
                if isinstance(value, str):
                    all_text_snippets.append(value)
                elif isinstance(value, list): # If a key contains a list of strings
                    for item_in_list in value:
                        if isinstance(item_in_list, str):
                            all_text_snippets.append(item_in_list)
                        # Could add recursion here if lists can contain more dicts:
                        # elif isinstance(item_in_list, (dict, list)):
                        #     all_text_snippets.extend(extract_text_from_json_data(item_in_list, text_keys))
    elif isinstance(data, list):
        for item in data:
            # Recursively call for items in the list, assuming they might be dicts or other lists
            all_text_snippets.extend(extract_text_from_json_data(item, text_keys))
    return all_text_snippets


def process_json_corpus(json_filepath, text_keys, use_indic_nlp_tokenizer=True):
    """
    Processes a JSON file to count word frequencies from specified text keys.
    """
    word_counts = Counter()

    print(f"Reading JSON file: {json_filepath}")
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
    except FileNotFoundError:
        print(f"Error: JSON file '{json_filepath}' not found.")
        return word_counts
    except json.JSONDecodeError as e:
        print(f"Error: Could not decode JSON from file '{json_filepath}'. {e}")
        return word_counts
    except Exception as e:
        print(f"Error reading JSON file {json_filepath}: {e}")
        return word_counts

    print(f"Extracting text from JSON keys: {', '.join(text_keys)}")
    
    all_text_snippets = extract_text_from_json_data(json_data, text_keys)

    if not all_text_snippets:
        print("No text found for the specified keys in the JSON file.")
        return word_counts

    print(f"Processing {len(all_text_snippets)} text snippets found in JSON...")
    
    iterable_snippets = tqdm(all_text_snippets, desc="Processing text snippets") if TQDM_AVAILABLE else all_text_snippets
    
    for text_snippet in iterable_snippets:
        if not text_snippet or not isinstance(text_snippet, str):
            continue # Skip if snippet is empty or not a string
        
        cleaned_text = clean_kannada_text(text_snippet)
        tokens = tokenize_kannada(cleaned_text, use_indic_nlp=use_indic_nlp_tokenizer)
        
        # Filter out very short tokens if desired. For Kannada, single characters can be meaningful.
        # Adjust min_len as needed, or remove this filter if all tokens are desired.
        min_token_len = 1 
        valid_tokens = [token for token in tokens if len(token) >= min_token_len and token.strip()] # ensure token is not just spaces
        word_counts.update(valid_tokens)
        
    return word_counts

def save_frequencies_to_db(db_path, word_counts):
    """
    Saves the calculated word frequencies to an SQLite database.
    Uses INSERT OR REPLACE to update frequencies if words already exist.
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        cursor.execute("""
        CREATE TABLE IF NOT EXISTS word_frequencies (
            word TEXT PRIMARY KEY,
            frequency INTEGER NOT NULL
        )
        """)
        print(f"Table 'word_frequencies' created or already exists in '{db_path}'.")

        data_to_insert = list(word_counts.items())
        
        if not data_to_insert:
            print("No word frequencies to save.")
            return

        print(f"Inserting/Updating {len(data_to_insert)} word frequencies into the database...")
        # Using INSERT OR REPLACE to update frequency if word exists, or insert if new.
        cursor.executemany("INSERT OR REPLACE INTO word_frequencies (word, frequency) VALUES (?, ?)", data_to_insert)
        
        conn.commit()
        print("Word frequencies saved to database successfully.")

    except sqlite3.Error as e:
        print(f"SQLite error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving to DB: {e}")
    finally:
        if conn:
            conn.close()

def main():
    parser = argparse.ArgumentParser(description="Calculate Kannada word frequencies from a JSON corpus and store them in an SQLite database.")
    parser.add_argument("json_file_path", help="Path to the JSON corpus file.")
    parser.add_argument("db_path", help="Path to the SQLite database file to create/update (e.g., kannada_frequencies.db).")
    parser.add_argument("--keys", nargs='+', required=True, 
                        help="One or more keys in the JSON objects from which to extract Kannada text (e.g., prompt answer text_field).")
    parser.add_argument("--no_indicnlp", action="store_true", 
                        help="Disable usage of indic-nlp-library for tokenization (uses basic regex fallback).")
    
    args = parser.parse_args()

    if not os.path.isfile(args.json_file_path):
        print(f"Error: JSON corpus file '{args.json_file_path}' not found or is not a file.")
        return

    use_indic_nlp_flag = not args.no_indicnlp
    if use_indic_nlp_flag and not INDIC_NLP_AVAILABLE:
        print("User requested Indic NLP for tokenization, but the library is not available. Falling back to basic regex tokenizer.")
        use_indic_nlp_flag = False 

    print(f"Starting word frequency calculation for JSON corpus at: {args.json_file_path}")
    word_counts = process_json_corpus(args.json_file_path, args.keys, use_indic_nlp_tokenizer=use_indic_nlp_flag)

    if word_counts:
        print(f"Total unique words found: {len(word_counts)}")
        # You can uncomment this to see the top words in your console:
        # print("Top 20 most common words:")
        # for word, count in word_counts.most_common(20):
        #     print(f"{word}: {count}")
        save_frequencies_to_db(args.db_path, word_counts)
    else:
        print("No words were processed or no frequencies were calculated. Check your JSON structure and keys.")

if __name__ == "__main__":
    main()