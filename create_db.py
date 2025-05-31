import sqlite3
import yaml
import json # For loading YAML if it has complex structures, not for storing stemmed words
import os
import re
from nltk.stem.snowball import SnowballStemmer # For stemming definition words
import nltk # For tokenization and stopword list

# --- Configuration ---
YAML_FILE_PATH = "alar_stemmed.yml" # Your YAML file
DB_PATH = "alar_corpus.db"         # Output SQLite database

# --- NLTK Setup for Stemming and Tokenization ---
# Ensure NLTK data is available. It's often better to run these downloads
# once in a Python interpreter if the script has issues in some environments.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK 'punkt' tokenizer models not found. Downloading...")
    nltk.download('punkt', quiet=True)
try:
    from nltk.corpus import stopwords
    stopwords.words('english') # Check if stopwords for English are available
except LookupError:
    print("NLTK 'stopwords' corpus not found. Downloading...")
    nltk.download('stopwords', quiet=True)


stemmer = SnowballStemmer("english")
# Using NLTK's default English stopword list
try:
    english_stopwords = set(stopwords.words('english'))
except Exception as e:
    print(f"Could not load NLTK stopwords, proceeding without stopword removal for stemmed_words_text: {e}")
    english_stopwords = set()


def create_tables(conn):
    """
    Creates database tables: 'words' and 'definitions'.
    'definitions' table stores raw text and a space-separated string of stemmed words.
    """
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS words (
        id INTEGER PRIMARY KEY,
        head TEXT,
        entry_text TEXT,
        phone TEXT,
        origin TEXT,
        info TEXT
    )
    """)
    print("Table 'words' created or already exists.")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS definitions (
        id INTEGER PRIMARY KEY,
        word_id INTEGER,
        definition_entry TEXT,       -- Raw definition text
        type TEXT,
        stemmed_words_text TEXT,     -- Space-separated stemmed words from definition_entry
        FOREIGN KEY (word_id) REFERENCES words (id)
    )
    """)
    print("Table 'definitions' (with definition_entry and stemmed_words_text) created or already exists.")

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_definitions_type ON definitions (type)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_definitions_word_id ON definitions (word_id)")
    print("Indexes created or already exist.")
    conn.commit()

def preprocess_text_for_stemmed_list(text):
    """
    Tokenizes, removes stopwords, and stems text.
    Returns a list of unique, sorted, stemmed words.
    """
    if not text or not isinstance(text, str):
        return []
    # Remove punctuation (except hyphens within words, though simple regex here removes all)
    cleaned_text = re.sub(r'[^\w\s-]', '', text.lower())
    try:
        tokens = nltk.word_tokenize(cleaned_text)
    except Exception as e:
        print(f"NLTK word_tokenize failed for text: '{text[:50]}...'. Error: {e}. Returning empty list.")
        return []
    
    stemmed_tokens = [
        stemmer.stem(token) for token in tokens 
        if token.isalnum() and token not in english_stopwords and len(token) > 1
    ]
    return sorted(list(set(stemmed_tokens)))


def populate_db_from_yaml(conn, yaml_file_path):
    """
    Loads data from YAML and populates the SQLite database.
    Generates a space-separated string of stemmed words for each definition.
    """
    print(f"Attempting to load YAML data from: '{yaml_file_path}'")
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as stream:
            word_list_yaml = yaml.safe_load(stream)
    except FileNotFoundError:
        print(f"ERROR: YAML file '{yaml_file_path}' not found.")
        return False
    except yaml.YAMLError as exc:
        print(f"ERROR: Could not parse YAML from file '{yaml_file_path}': {exc}")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading YAML data: {e}")
        return False

    if not isinstance(word_list_yaml, list):
        print(f"ERROR: Data in '{yaml_file_path}' is not a list at the root level.")
        return False
    
    print(f"Successfully loaded {len(word_list_yaml)} items from YAML file.")
    print("Processing definitions and populating database...")

    cursor = conn.cursor()
    entries_added_count = 0
    definitions_added_count = 0
    skipped_items_count = 0
    skipped_definitions_count = 0

    for item_index, item_data in enumerate(word_list_yaml):
        # Validate main item structure
        if not (isinstance(item_data, dict) and 'id' in item_data and 
                'entry' in item_data and 'defs' in item_data and 
                isinstance(item_data.get('defs'), list)):
            print(f"Warning: Skipping item at index {item_index} (ID: {item_data.get('id', 'N/A')}) - invalid structure.")
            skipped_items_count += 1
            continue
        
        word_main_id = item_data.get('id')
        try:
            cursor.execute("INSERT OR IGNORE INTO words (id, head, entry_text, phone, origin, info) VALUES (?, ?, ?, ?, ?, ?)",
                           (word_main_id, item_data.get('head'), item_data.get('entry'), item_data.get('phone'), item_data.get('origin'), item_data.get('info')))
            if cursor.rowcount > 0: entries_added_count += 1
        except sqlite3.Error as e:
            print(f"ERROR: SQLite (words table) for word ID {word_main_id}: {e}")
            skipped_items_count +=1; continue

        for def_index, definition_data in enumerate(item_data.get('defs', [])):
            # Validate definition structure
            if not (isinstance(definition_data, dict) and 'id' in definition_data and 
                    'entry' in definition_data and 'type' in definition_data):
                print(f"Warning: Skipping definition for word ID {word_main_id} (Def ID: {definition_data.get('id', 'N/A')}) - missing 'id', 'entry', or 'type'.")
                skipped_definitions_count += 1; continue

            definition_id = definition_data.get('id')
            definition_text_raw = definition_data.get('entry') 
            
            stemmed_words_list = []
            if definition_text_raw and isinstance(definition_text_raw, str):
                stemmed_words_list = preprocess_text_for_stemmed_list(definition_text_raw)
            
            stemmed_text_content = ' '.join(stemmed_words_list) if stemmed_words_list else None

            try:
                cursor.execute("""
                INSERT OR IGNORE INTO definitions (id, word_id, definition_entry, type, stemmed_words_text)
                VALUES (?, ?, ?, ?, ?)
                """, (definition_id, word_main_id, definition_text_raw, definition_data.get('type'), stemmed_text_content))
                if cursor.rowcount > 0: definitions_added_count += 1
            except sqlite3.Error as e:
                 print(f"ERROR: SQLite (definitions table) for Def ID {definition_id}, Word ID {word_main_id}: {e}")
                 skipped_definitions_count +=1
    
    conn.commit()
    print(f"\n--- Database Population Summary ---")
    print(f"New main word entries added: {entries_added_count}")
    print(f"New definitions processed: {definitions_added_count}")
    if skipped_items_count > 0: print(f"Skipped main items: {skipped_items_count}")
    if skipped_definitions_count > 0: print(f"Skipped definitions: {skipped_definitions_count}")
    print("---------------------------------")
    return True

def main():
    print(f"Database will be created/updated at: '{DB_PATH}'")
    
    if not os.path.exists(YAML_FILE_PATH):
        print(f"CRITICAL ERROR: YAML input file '{YAML_FILE_PATH}' not found.")
        return

    db_connection = None
    try:
        db_connection = sqlite3.connect(DB_PATH)
        print(f"Successfully connected to (or created) database: '{DB_PATH}'.")
        
        cursor = db_connection.cursor()
        NEEDS_RECREATE = False
        try: # Check if definitions table exists and if it has old/unwanted columns
            cursor.execute("PRAGMA table_info(definitions)")
            columns = [info[1] for info in cursor.fetchall()]
            if 'definition_embedding' in columns or 'stemmed_words_list_json' in columns :
                print("\nWARNING: Old database schema (with embeddings or JSON stemmed words) detected.")
                NEEDS_RECREATE = True
            
            # Check if definitions table exists before checking its columns specifically for stemmed_words_text
            definitions_table_exists = any(tbl[0] == 'definitions' for tbl in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
            if definitions_table_exists and 'stemmed_words_text' not in columns:
                 print("\nWARNING: 'stemmed_words_text' column missing from existing definitions table.")
                 NEEDS_RECREATE = True
        except sqlite3.OperationalError: 
            pass # Table might not exist yet, create_tables will handle it

        if NEEDS_RECREATE:
            user_choice_recreate = input("Do you want to drop existing tables and recreate with the new simplified schema? (yes/no): ").strip().lower()
            if user_choice_recreate == 'yes':
                print("Dropping existing tables...")
                cursor.execute("DROP TABLE IF EXISTS definitions")
                cursor.execute("DROP TABLE IF EXISTS words")
                db_connection.commit()
                print("Tables dropped.")
            else:
                print("Exiting. Please manually update schema or allow recreation.")
                db_connection.close()
                return
        
        create_tables(db_connection) 
        
        cursor.execute("SELECT COUNT(id) FROM words") 
        word_count = cursor.fetchone()[0]
        
        if word_count > 0 and not NEEDS_RECREATE: 
            print(f"\nDatabase already contains {word_count} word entries.")
            print("Re-running will add new entries if their IDs are not already in the database.")
            user_choice_populate = input("Proceed with populating/updating from YAML? (yes/no): ").strip().lower()
            if user_choice_populate == 'yes':
                populate_db_from_yaml(db_connection, YAML_FILE_PATH)
            else:
                print("Skipping population step as per user choice.")
        else:
            print("\nDatabase appears to be empty or was just recreated. Proceeding with initial population...")
            populate_db_from_yaml(db_connection, YAML_FILE_PATH)

    except sqlite3.Error as e:
        print(f"A SQLite error occurred: {e}")
    except Exception as e:
        print(f"An unexpected general error occurred: {e}")
    finally:
        if db_connection:
            db_connection.close()
            print("Database connection closed.")

if __name__ == "__main__":
    main()