import sqlite3
import yaml
import json
import os
import re # Included for consistency, though not strictly used in this version

# --- Configuration ---
# Default name for the YAML input file.
# Ensure this file is in the same directory as the script, or provide the full path.
YAML_FILE_PATH = "alar_stemmed.yml"
# Default name for the SQLite database file that will be created.
DB_PATH = "alar_corpus.db"

def create_tables(conn):
    """
    Creates the necessary database tables ('words' and 'definitions')
    if they do not already exist.
    Includes indexes for faster lookups.
    """
    cursor = conn.cursor()
    # --- words Table ---
    # Stores the main dictionary entries.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS words (
        id INTEGER PRIMARY KEY,      -- Unique ID for the word entry (from YAML)
        head TEXT,                   -- Headword, often in the primary script
        entry_text TEXT,             -- The main lexical entry text
        phone TEXT,                  -- Phonetic transcription
        origin TEXT,                 -- Word origin or etymology
        info TEXT                    -- Miscellaneous information
    )
    """)
    print("Table 'words' created or already exists.")

    # --- definitions Table ---
    # Stores individual definitions associated with entries in the 'words' table.
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS definitions (
        id INTEGER PRIMARY KEY,             -- Unique ID for the definition (from YAML defs section)
        word_id INTEGER,                  -- Foreign key linking to words.id
        definition_entry TEXT,            -- The text of the definition
        type TEXT,                        -- Part of speech for this definition
        stemmed_words_list_json TEXT,     -- JSON string of stemmed words from definition_entry
        FOREIGN KEY (word_id) REFERENCES words (id)
    )
    """)
    print("Table 'definitions' created or already exists.")

    # --- Indexes ---
    # Index on 'type' in definitions for faster filtering by part of speech.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_definitions_type ON definitions (type)")
    # Index on 'word_id' in definitions for faster joining/lookup of definitions for a word.
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_definitions_word_id ON definitions (word_id)")
    print("Indexes created or already exist.")

    conn.commit()

def populate_db_from_yaml(conn, yaml_file_path):
    """
    Loads data from the specified YAML file and populates the SQLite database.
    It inserts data into the 'words' and 'definitions' tables.
    Uses 'INSERT OR IGNORE' to avoid errors if an ID already exists,
    effectively allowing the script to be re-run without duplicating entries
    (though it won't update existing entries with the same ID).
    """
    print(f"Attempting to load YAML data from: '{yaml_file_path}'")
    try:
        with open(yaml_file_path, "r", encoding="utf-8") as stream:
            word_list_yaml = yaml.safe_load(stream)
    except FileNotFoundError:
        print(f"ERROR: YAML file '{yaml_file_path}' not found. Cannot populate database.")
        return False
    except yaml.YAMLError as exc:
        print(f"ERROR: Could not parse YAML from file '{yaml_file_path}': {exc}")
        return False
    except Exception as e:
        print(f"ERROR: An unexpected error occurred while loading YAML data: {e}")
        return False

    if not isinstance(word_list_yaml, list):
        print(f"ERROR: Data in '{yaml_file_path}' is not a list at the root level. Expected a list of dictionary entries.")
        return False
    
    print(f"Successfully loaded {len(word_list_yaml)} items from YAML file.")

    cursor = conn.cursor()
    entries_added_count = 0
    definitions_added_count = 0
    skipped_items_count = 0
    skipped_definitions_count = 0

    for item_index, item_data in enumerate(word_list_yaml):
        # Basic validation for the main item structure
        if not (isinstance(item_data, dict) and
                'id' in item_data and
                'entry' in item_data and # Main word entry
                'defs' in item_data and isinstance(item_data.get('defs'), list)): # Defs list
            print(f"Warning: Skipping item at index {item_index} (ID: {item_data.get('id', 'N/A')}) due to missing required keys ('id', 'entry', 'defs') or incorrect 'defs' format.")
            skipped_items_count += 1
            continue
        
        word_main_id = item_data.get('id')

        try:
            # Insert into 'words' table
            cursor.execute("""
            INSERT OR IGNORE INTO words (id, head, entry_text, phone, origin, info)
            VALUES (?, ?, ?, ?, ?, ?)
            """, (
                word_main_id,
                item_data.get('head'),
                item_data.get('entry'),
                item_data.get('phone'),
                item_data.get('origin'),
                item_data.get('info')
            ))
            if cursor.rowcount > 0: # rowcount is 1 if a new row was inserted
                entries_added_count += 1
        except sqlite3.Error as e:
            print(f"ERROR: SQLite error inserting word ID {word_main_id}: {e}")
            skipped_items_count += 1
            continue # Skip definitions if word insertion failed
        except Exception as e:
            print(f"ERROR: Unexpected error processing word ID {word_main_id}: {e}")
            skipped_items_count += 1
            continue

        # Process definitions for the current word item
        for def_index, definition_data in enumerate(item_data.get('defs', [])):
            # Basic validation for definition structure
            if not (isinstance(definition_data, dict) and
                    'id' in definition_data and
                    'entry' in definition_data and # Definition text
                    'type' in definition_data and # Part of speech
                    '_stemmed_words_list' in definition_data and
                    isinstance(definition_data.get('_stemmed_words_list'), list)):
                print(f"Warning: Skipping definition at index {def_index} for word ID {word_main_id} (Def ID: {definition_data.get('id', 'N/A')}) due to missing keys or incorrect '_stemmed_words_list' format.")
                skipped_definitions_count += 1
                continue

            definition_id = definition_data.get('id')
            stemmed_list = definition_data.get('_stemmed_words_list', [])
            
            # Ensure all elements in stemmed_list are strings before JSON serialization
            if not all(isinstance(s, str) for s in stemmed_list):
                print(f"Warning: Skipping definition ID {definition_id} for word ID {word_main_id}. '_stemmed_words_list' contains non-string elements. Data: {stemmed_list}")
                skipped_definitions_count += 1
                continue
            
            stemmed_list_json = json.dumps(stemmed_list) # Convert list to JSON string

            try:
                # Insert into 'definitions' table
                cursor.execute("""
                INSERT OR IGNORE INTO definitions (id, word_id, definition_entry, type, stemmed_words_list_json)
                VALUES (?, ?, ?, ?, ?)
                """, (
                    definition_id,
                    word_main_id, # Foreign key linking to the 'words' table
                    definition_data.get('entry'),
                    definition_data.get('type'),
                    stemmed_list_json
                ))
                if cursor.rowcount > 0:
                    definitions_added_count += 1
            except sqlite3.Error as e:
                 print(f"ERROR: SQLite error inserting definition ID {definition_id} for word ID {word_main_id}: {e}")
                 skipped_definitions_count +=1
            except Exception as e:
                print(f"ERROR: Unexpected error processing definition ID {definition_id} for word ID {word_main_id}: {e}")
                skipped_definitions_count += 1

    conn.commit() # Commit all transactions
    print(f"\n--- Database Population Summary ---")
    print(f"New main word entries added: {entries_added_count}")
    print(f"New definitions added: {definitions_added_count}")
    if skipped_items_count > 0:
        print(f"Skipped main items: {skipped_items_count}")
    if skipped_definitions_count > 0:
        print(f"Skipped definitions: {skipped_definitions_count}")
    print("---------------------------------")
    return True

def main():
    """
    Main function to orchestrate database creation and population.
    """
    print(f"Database will be created/updated at: '{DB_PATH}'")
    
    if not os.path.exists(YAML_FILE_PATH):
        print(f"CRITICAL ERROR: YAML input file '{YAML_FILE_PATH}' not found.")
        print("Please ensure the YAML file is in the same directory as this script, or update the YAML_FILE_PATH variable.")
        return

    db_connection = None
    try:
        # Connect to SQLite database (creates the file if it doesn't exist)
        db_connection = sqlite3.connect(DB_PATH)
        print(f"Successfully connected to (or created) database: '{DB_PATH}'.")

        create_tables(db_connection) # Ensure tables are created

        # Check if database already has data to avoid accidental re-runs or to prompt user
        cursor = db_connection.cursor()
        cursor.execute("SELECT COUNT(id) FROM words")
        word_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(id) FROM definitions")
        def_count = cursor.fetchone()[0]

        if word_count > 0 or def_count > 0:
            print(f"\nDatabase already contains data: {word_count} words, {def_count} definitions.")
            # It's safe to re-run due to "INSERT OR IGNORE", it won't duplicate existing IDs.
            # It will add any new items from YAML not already in DB.
            # If you need to fully refresh (delete old, insert new), you'd need to drop tables first.
            print("Re-running will add new entries from YAML if their IDs are not already in the database.")
            print("Existing entries with the same ID will NOT be updated by this script.")
            user_choice = input("Do you want to proceed with populating/updating from YAML? (yes/no): ").strip().lower()
            if user_choice == 'yes':
                populate_db_from_yaml(db_connection, YAML_FILE_PATH)
            else:
                print("Skipping population step as per user choice.")
        else:
            print("\nDatabase appears to be empty. Proceeding with initial population...")
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
    # This block executes when the script is run directly (e.g., `python create_db.py`)
    main()
