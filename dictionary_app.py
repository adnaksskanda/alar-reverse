import streamlit as st
import sqlite3
import json
import os
import re
from nltk.stem.snowball import SnowballStemmer

# --- 0. Global Configuration & Stemmer ---
DB_PATH = "alar_corpus.db"  # Database created by create_db.py
stemmer = SnowballStemmer("english")

# --- 1. Database Connection and Initialization ---

@st.cache_resource
def get_db_connection(db_path):
    """
    Connects to the SQLite database in read-only mode.
    Returns a connection object or None if connection fails.
    Caches the connection resource for efficiency.
    """
    try:
        # Connect in read-only mode ('ro') as the Streamlit app should not modify the DB.
        # uri=True is needed for mode parameter in some sqlite versions/OS.
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row  # Access columns by name
        return conn
    except sqlite3.OperationalError as e:
        # This error can occur if the file doesn't exist or if read-only mode is problematic
        # (e.g. if the DB file itself is missing, which is checked before this).
        st.error(f"Error connecting to database '{db_path}' in read-only mode: {e}")
        st.error("Ensure the database file exists and the application has read permissions.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while connecting to the database: {e}")
        return None

def check_database_tables_exist(conn):
    """Checks if the required tables 'words' and 'definitions' exist in the DB."""
    if not conn:
        return False
    try:
        cursor = conn.cursor()
        # Check for 'words' table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='words'")
        if not cursor.fetchone():
            st.error("Database error: 'words' table is missing.")
            return False
        # Check for 'definitions' table
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='definitions'")
        if not cursor.fetchone():
            st.error("Database error: 'definitions' table is missing.")
            return False
        return True
    except sqlite3.Error as e:
        st.error(f"SQLite error while checking database tables: {e}")
        return False

def initialize_database_connection():
    """
    Checks for DB file existence, attempts to connect, and verifies table integrity.
    Returns a connection object or None if setup is incomplete, guiding the user.
    """
    if not os.path.exists(DB_PATH):
        st.error(f"Database file '{DB_PATH}' not found!")
        st.info("Create database, then run this app.")
        return None

    conn = get_db_connection(DB_PATH)
    if conn is None:
        # get_db_connection would have already shown an error.
        # Add a general message if it's still None.
        st.error("Failed to establish a database connection. Please check previous error messages.")
        return None

    if not check_database_tables_exist(conn):
        st.error(
            f"The database '{DB_PATH}' exists, but the required tables ('words', 'definitions') "
            f"are missing or could not be verified."
        )
        st.info(
            "The database might be corrupted or was not fully created. "
            "Please try running the `create_db.py` script again."
        )
        # It's good practice to close a connection if it's problematic or not going to be used.
        try:
            conn.close()
        except sqlite3.Error:
            pass # Ignore errors if closing fails on a bad connection
        return None
    
    return conn

# --- 2. Data Access Functions (Fetching from Database) ---

@st.cache_data # Cache the result of this function
def get_total_entries_count(_conn):
    """Gets the total number of word entries from the \'words\' table."""
    # _conn argument is used by Streamlit to track the resource.
    if not _conn:
        return 0
    try:
        cursor = _conn.cursor()
        cursor.execute("SELECT COUNT(id) FROM words")
        count = cursor.fetchone()[0]
        return count
    except sqlite3.Error as e:
        st.error(f"Error fetching total entries count from database: {e}")
        return 0

@st.cache_data # Cache the result of this function
def get_unique_pos_list(_conn):
    """Gets a list of unique, non-empty parts of speech from the 'definitions' table."""
    if not _conn:
        return ["Any"]
    try:
        cursor = _conn.cursor()
        # Ensure type is not NULL and not an empty string for a cleaner list
        cursor.execute("SELECT DISTINCT type FROM definitions WHERE type IS NOT NULL AND type != '' ORDER BY type")
        # Fetchall returns a list of tuples, e.g., [('noun',), ('verb',)]
        pos_list = [row[0] for row in cursor.fetchall()]
        return ["Any"] + pos_list # Prepend "Any" for the filter
    except sqlite3.Error as e:
        st.error(f"Error fetching unique parts of speech from database: {e}")
        return ["Any"]


# --- 3. Search Function (SQLite version) ---

def search_corpus_from_db(conn, search_definition_query, selected_pos):
    """
    Searches entries based on text within definitions and filters by part of speech (type)
    using the SQLite database.
    - conn: Active SQLite database connection.
    - search_definition_query: Text to search in the 'entry' field of definitions.
    - selected_pos: Part of speech to filter by. 'Any' means no POS filter.
    """
    if not conn:
        st.error("Database connection is not available for search.")
        return []

    search_def_lower = search_definition_query.lower().strip() if search_definition_query else ""
    # Normalize "Any" POS selection for SQL and logic
    is_pos_filter_active = selected_pos and selected_pos.lower() != "any"
    selected_pos_lower = selected_pos.lower() if is_pos_filter_active else ""

    is_text_query_active = bool(search_def_lower)
    stemmed_query_words = set()

    if is_text_query_active:
        cleaned_query_text = re.sub(r'[^\w\s-]', '', search_def_lower) # Remove punctuation except hyphen
        query_words = set(word for word in cleaned_query_text.split() if word) # Tokenize and remove empty strings
        if not query_words: # If query was only punctuation or empty
            is_text_query_active = False # No valid text to search
        else:
            stemmed_query_words = set(stemmer.stem(word) for word in query_words)
            if not stemmed_query_words: # If stemming resulted in empty set (unlikely with valid words)
                is_text_query_active = False
    
    # If neither filter is active, return empty list
    if not is_text_query_active and not is_pos_filter_active:
        return []

    matching_word_ids = set()
    cursor = conn.cursor()

    # Step 1: Identify candidate definitions based on POS filter (if active).
    # Then, filter these candidates by text query in Python.
    sql_query_definitions = "SELECT word_id, type, stemmed_words_list_json FROM definitions"
    params = []
    
    if is_pos_filter_active:
        sql_query_definitions += " WHERE LOWER(type) = ?"
        params.append(selected_pos_lower)

    try:
        cursor.execute(sql_query_definitions, params)
        candidate_definitions = cursor.fetchall() # List of sqlite3.Row objects
    except sqlite3.Error as e:
        st.error(f"Error querying definitions from database: {e}")
        return []

    # Step 2: Filter candidate definitions by text query (if active)
    for def_row in candidate_definitions:
        current_word_id = def_row['word_id']
        
        # If text query is not active, this definition (already POS-filtered by SQL) is a match for its word_id.
        passes_text_filter = not is_text_query_active
        
        if is_text_query_active:
            stemmed_json_from_db = def_row['stemmed_words_list_json']
            if stemmed_json_from_db:
                try:
                    # The stemmed words are stored as a JSON list in the DB
                    stemmed_definition_words = set(json.loads(stemmed_json_from_db))
                    if stemmed_query_words.issubset(stemmed_definition_words):
                        passes_text_filter = True
                except json.JSONDecodeError:
                    # Log or handle malformed JSON if necessary, treat as no match
                    # print(f"Warning: Malformed JSON for stemmed_words_list in def for word_id {current_word_id}")
                    passes_text_filter = False 
            else: # No stemmed words stored for this definition
                passes_text_filter = False
        
        if passes_text_filter:
            matching_word_ids.add(current_word_id)
            if len(matching_word_ids) > 200: # Optimization: stop collecting if too many distinct words match
                break


    if not matching_word_ids:
        return []

    # Step 3: Fetch full word data and all their definitions for the matching_word_ids
    word_ids_list_for_query = list(matching_word_ids)
    placeholders = ','.join(['?'] * len(word_ids_list_for_query)) # e.g., "?,?,?"

    # Fetch main word details
    sql_fetch_words = f"SELECT id, head, entry_text, phone, origin, info FROM words WHERE id IN ({placeholders})"
    try:
        cursor.execute(sql_fetch_words, word_ids_list_for_query)
        word_rows = cursor.fetchall()
        # Create a dictionary for quick lookup: {word_id: word_data_dict}
        words_data_map = {row['id']: dict(row) for row in word_rows}
    except sqlite3.Error as e:
        st.error(f"Error fetching word details from database: {e}")
        return []

    # Fetch all definitions for these matched words (for display purposes)
    sql_fetch_all_definitions = f"SELECT id, word_id, definition_entry, type FROM definitions WHERE word_id IN ({placeholders})"
    try:
        cursor.execute(sql_fetch_all_definitions, word_ids_list_for_query)
        all_definitions_for_matched_words = cursor.fetchall()
    except sqlite3.Error as e:
        st.error(f"Error fetching all definitions for matched words: {e}")
        return []
    
    # Group definitions by their word_id
    definitions_by_word_id_map = {}
    for def_row in all_definitions_for_matched_words:
        w_id = def_row['word_id']
        if w_id not in definitions_by_word_id_map:
            definitions_by_word_id_map[w_id] = []
        # Structure each definition as a dictionary, similar to original YAML structure
        definitions_by_word_id_map[w_id].append({
            'id': def_row['id'], 
            'entry': def_row['definition_entry'],
            'type': def_row['type']
        })

    # Step 4: Assemble final results list, maintaining structure expected by UI
    final_results_list = []
    # Sort by word ID to have a somewhat consistent order, similar to original file order
    # This uses the word_ids that actually matched, not necessarily all from word_ids_list_for_query if some were filtered out
    sorted_matched_word_ids = sorted(list(matching_word_ids)) 

    for word_id_val in sorted_matched_word_ids:
        if word_id_val not in words_data_map: # Should not happen if logic is correct
            continue

        word_info_dict = words_data_map[word_id_val]
        # Reconstruct the item structure to match what the UI expects
        # This structure is similar to the original `word_dict_lst` items
        result_item = {
            'id': word_info_dict['id'], 
            'entry': word_info_dict['entry_text'], # Main entry text for the word
            'head': word_info_dict['head'],
            'phone': word_info_dict['phone'],
            'origin': word_info_dict['origin'],
            'info': word_info_dict['info'],
            'defs': definitions_by_word_id_map.get(word_id_val, []) # List of definition dicts
        }
        final_results_list.append(result_item)
        if len(final_results_list) >= 101: # Apply result limit (original script was >100)
            break
            
    return final_results_list


# --- 4. Streamlit User Interface ---

st.set_page_config(layout="wide") # Use wide layout for more space
st.title("📖 [alar.ink](https://alar.ink/) Corpus English Definition Search (SQLite)")
st.markdown("""
Search within [alar.ink](https://alar.ink/)'s English definitions for a matching word in Kannada and filter by part of speech (type).
This version uses an SQLite database (`alar_corpus.db`) for data storage and querying.
Please ensure `create_db.py` has been run to generate the database from your YAML source.
""")

# Attempt to initialize database connection.
# This will show errors and instructions if DB is not set up correctly.
db_connection = initialize_database_connection()

if db_connection is None:
    st.warning("Application cannot proceed until the database is correctly set up. Please follow the instructions above.")
    st.stop() # Halt further execution of the Streamlit app if DB connection fails

# --- Proceed only if db_connection is valid ---
total_entries_in_db = get_total_entries_count(db_connection)
unique_pos_list_from_db = get_unique_pos_list(db_connection)

if total_entries_in_db == 0 :
    st.error(f"The database '{DB_PATH}' appears to be empty (no word entries found).")
    st.info(
        f"Please ensure the `create_db.py` script was run successfully and populated "
        f"the database with data from your YAML file (e.g., 'alar_stemmed.yml')."
    )
    st.stop()


# --- UI Layout for Search Inputs ---
col1, col2 = st.columns([3, 2]) 

with col1:
    search_definition_term_input = st.text_input(
        "Search within definitions:", 
        placeholder="e.g., hint, rice, sweetness"
    )
with col2:
    selected_part_of_speech_filter = st.selectbox(
        "Filter by Part of Speech (type):", 
        unique_pos_list_from_db # Populated from DB
    )

# --- Perform Search and Display Results ---
# Trigger search if there's a search term OR a POS filter (other than "Any") is selected
if search_definition_term_input or (selected_part_of_speech_filter and selected_part_of_speech_filter != "Any"):
    st.markdown("---") 
    
    # Call the database search function
    search_results_list = search_corpus_from_db(
        db_connection, 
        search_definition_term_input, 
        selected_part_of_speech_filter
    )
    total_found_count = len(search_results_list)

    if search_results_list:
        # Build the subheader message dynamically based on active filters
        subheader_message_parts = []
        if search_definition_term_input:
            subheader_message_parts.append(f"definitions matching \"{search_definition_term_input}\"")
        if selected_part_of_speech_filter and selected_part_of_speech_filter != "Any":
            subheader_message_parts.append(f"part of speech \"{selected_part_of_speech_filter.capitalize()}\"")
        
        criteria_message = " and ".join(subheader_message_parts)
        if not criteria_message: # Should not happen if this block is reached
            criteria_message = "your criteria"

        st.subheader(f"Found {total_found_count} entr{'y' if total_found_count == 1 else 'ies'} where {criteria_message}:")
        
        if total_found_count > 100: # Check if it hit the display limit (original script was >100, so 101 is the limit)
            st.info("Search may have been too broad; displaying the first 101 matching entries.")

        # Display each result entry
        for i, entry_data_dict in enumerate(search_results_list): # entry_data_dict is a dict from search_corpus_from_db
            display_word = entry_data_dict.get('entry', "Unknown Entry") # Main word entry

            # Determine Part of Speech for the expander title (logic similar to original script)
            expander_pos_display = "N/A" 
            current_definitions_list = entry_data_dict.get("defs", []) # List of definition dicts
            if current_definitions_list:
                if selected_part_of_speech_filter and selected_part_of_speech_filter != "Any":
                    # Check if any definition in this entry matches the selected POS for display consistency
                    if any(d.get("type", "").strip().lower() == selected_part_of_speech_filter.lower() for d in current_definitions_list):
                        expander_pos_display = selected_part_of_speech_filter.capitalize()
                    # Fallback to the first definition's type if no direct match for the filter (e.g. text matched but POS was different)
                    elif current_definitions_list[0].get("type"): 
                         expander_pos_display = current_definitions_list[0].get("type")
                # If no POS filter was active, use the type of the first definition for the expander
                elif current_definitions_list[0].get("type"): 
                    expander_pos_display = current_definitions_list[0].get("type")
            
            with st.expander(f"{display_word} ({expander_pos_display})", expanded=(i == 0)): # Expand first result by default
                # Display other word details
                if entry_data_dict.get('phone'):
                    st.markdown(f"**Phonetic:** {entry_data_dict.get('phone')}")
                if entry_data_dict.get('origin'):
                    st.markdown(f"**Origin:** {entry_data_dict.get('origin')}")
                if entry_data_dict.get('info'):
                    st.markdown(f"**Info:** {entry_data_dict.get('info')}")
                
                st.markdown("**Definitions:**")
                if current_definitions_list:
                    for def_idx, definition_dict in enumerate(current_definitions_list):
                        def_text = definition_dict.get('entry', 'No definition entry.')
                        def_type = definition_dict.get('type', 'N/A')
                        st.markdown(f"- **{def_text}** (*{def_type}*)")
                        # Add a separator between definitions if there are multiple
                        if def_idx < len(current_definitions_list) - 1: 
                            st.markdown("---") 
                else:
                    st.write("No definitions listed for this entry.")
    
    # If search was triggered but no results found
    elif search_definition_term_input or (selected_part_of_speech_filter and selected_part_of_speech_filter != "Any"): 
        st.info(f"No entries found matching your criteria.")
else:
    # Initial state or when search term is cleared and POS is "Any"
    st.info("Enter a term to search within definitions and/or select a part of speech to filter.")

# --- Sidebar Information ---
st.sidebar.header("About")
st.sidebar.info(
    "This application allows you to search within dictionary definitions "
    "and filter by part of speech (type).\n\n"
    "It loads data from the alar.ink corpus (via an SQLite database) "
    "created by V. Krishna and Kailash Nadh."
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"Total unique entries in database: **{total_entries_in_db}**")
st.sidebar.markdown("---")
st.sidebar.caption(f"Ensure '{DB_PATH}' is present and populated. If not, run `create_db.py`.")

# The Streamlit app automatically closes the connection when the script finishes or Streamlit re-runs.
# However, explicitly closing is good practice if you were managing connections manually for long-lived objects.
# For Streamlit's execution model with @st.cache_resource, direct manual closing here is not typically needed
# as Streamlit manages the lifecycle of cached resources.
# if db_connection:
#     db_connection.close()
