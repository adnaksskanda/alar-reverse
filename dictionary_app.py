import streamlit as st
import sqlite3
import os
import re
import numpy as np # For numerical operations, often used with scikit-learn

# NLTK imports
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# --- Attempt to load NLTK resources and set global variables ---
wn = None
english_stopwords_global = set()
wordnet_available = False

# --- Sentence Transformer Model Loading ---
# Placed after global configurations
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2' # A good default
SEMANTIC_SIMILARITY_THRESHOLD = 0.5      # Adjust for sentence embeddings

KANNADA_FREQUENCY_DB_PATH = "./kannada_frequencies.db" # <<< NEW: Path to your frequency DB
COMMONNESS_WEIGHT = 0.3  # <<< NEW: Weight for commonness score (0.0 to 1.0)
RELEVANCE_WEIGHT = 0.7   # <<< NEW: Weight for relevance (semantic similarity) score



# --- 0. Global Configuration & Components ---
DB_PATH = "alar_corpus.db"
SIMILARITY_THRESHOLD_TFIDF = 0.05 
MAX_PREFILTER_CANDIDATES = 1000   # As per your last version
MAX_FINAL_RESULTS = 101          

stemmer_global = SnowballStemmer("english")
lemmatizer_global = WordNetLemmatizer()


try:
    from nltk.corpus import wordnet as wn_imported
    from nltk.corpus import stopwords as stopwords_imported
    
    # Try to access the resources to ensure they are downloaded and working
    nltk.data.find('tokenizers/punkt')  # For word_tokenize
    nltk.data.find('taggers/averaged_perceptron_tagger') # For nltk.pos_tag
    
    wn = wn_imported # Assign to global wn
    english_stopwords_global = set(stopwords_imported.words('english'))
    
    # Test WordNet by trying to get a synset
    if wn:
        wn.synsets("test") # This will raise LookupError if wordnet or omw-1.4 is missing
        wordnet_available = True
        print("NLTK resources (WordNet, stopwords, punkt, averaged_perceptron_tagger) appear to be available.")
    else: # Should not happen if import succeeded, but as a safeguard
        wordnet_available = False
        print("nltk.corpus.wordnet could not be imported as 'wn'.")

except LookupError as e:
    st.error(
        f"NLTK LookupError: Required data package not found: {e}. "
        "The application might not function correctly. "
        "Please ensure 'punkt', 'stopwords', 'wordnet', 'omw-1.4', and 'averaged_perceptron_tagger' are downloaded. "
        "You can try running this in your Python interpreter (in the correct conda environment):\n\n"
        "import nltk\n"
        "nltk.download('punkt')\n"
        "nltk.download('stopwords')\n"
        "nltk.download('wordnet')\n"
        "nltk.download('omw-1.4')\n"
        "nltk.download('averaged_perceptron_tagger')\n"
    )
    wordnet_available = False # Ensure this is False if any critical resource is missing
except ImportError as e:
    st.error(f"NLTK ImportError: {e}. Please ensure NLTK is installed correctly.")
    wordnet_available = False
except Exception as e: # Catch any other unexpected error during NLTK setup
    st.error(f"An unexpected error occurred during NLTK setup: {e}")
    wordnet_available = False

try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    # You'll add a UI warning later if this is False and user tries to use it


@st.cache_resource
def load_sentence_embedding_model(model_name):
    if not sentence_transformers_available:
        # This message will appear in console, UI warnings handled later
        print("`sentence-transformers` library not installed. Semantic search disabled.")
        return None
    try:
        # This print will show in console during startup
        print(f"Loading sentence embedding model ({model_name})...")
        model = SentenceTransformer(model_name)
        print(f"Sentence embedding model ({model_name}) loaded.")
        return model
    except Exception as e:
        # This print will show in console
        print(f"Error loading sentence model '{model_name}': {e}")
        return None

sentence_model_global = load_sentence_embedding_model(SENTENCE_MODEL_NAME)

# ...


# --- Add these new functions, perhaps near your other DB functions ---

@st.cache_resource
def get_frequency_db_connection(db_path):
    """Connects to the Kannada word frequency SQLite database."""
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        # No row_factory needed if just fetching one value
        return conn
    except Exception as e:
        st.error(f"Error connecting to frequency database '{db_path}': {e}")
        return None

@st.cache_data # Cache individual frequency lookups
def get_kannada_word_frequency(_freq_db_conn, kannada_word):
    """Looks up the frequency of a Kannada word from the frequency database."""
    if not _freq_db_conn or not kannada_word:
        return 0 # Default frequency if word is None or DB connection fails
    
    try:
        cursor = _freq_db_conn.cursor()
        cursor.execute("SELECT frequency FROM word_frequencies WHERE word = ?", (kannada_word,))
        result = cursor.fetchone()
        return result[0] if result else 0 # Return frequency or 0 if word not found
    except sqlite3.Error as e:
        # This might be too noisy if many words are not found
        # st.warning(f"SQLite error looking up frequency for '{kannada_word}': {e}")
        return 0 # Default to 0 on error
    except Exception: # Catch any other error
        return 0


# --- 1. Database Connection and Initialization ---
@st.cache_resource
def get_db_connection(db_path):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.OperationalError as e:
        st.error(f"Error connecting to database '{db_path}' in read-only mode: {e}. Ensure file exists and has read permissions.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred connecting to database: {e}")
        return None

def check_database_tables_exist(conn): 
    if not conn: return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='words'")
        if not cursor.fetchone(): st.error("DB error: 'words' table missing."); return False
        
        cursor.execute("PRAGMA table_info(definitions)")
        columns = [info[1] for info in cursor.fetchall()]
        
        definitions_table_exists = any(tbl[0] == 'definitions' for tbl in cursor.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall())
        if not definitions_table_exists:
             st.error("DB error: 'definitions' table missing."); return False
        if 'stemmed_words_text' not in columns:
            st.error("DB error: 'stemmed_words_text' column missing. Ensure create_db.py (simplified version) was run.")
            return False
        if 'definition_entry' not in columns:
             st.error("DB error: 'definition_entry' column missing. Ensure create_db.py (simplified version) was run.")
             return False
        return True
    except sqlite3.Error as e: 
        st.error(f"SQLite error checking DB tables: {e}"); return False

def initialize_database_connection():
    if not os.path.exists(DB_PATH): 
        st.error(f"Database file '{DB_PATH}' not found! Please run the appropriate `create_db.py` script first to generate it from your YAML source."); return None
    conn = get_db_connection(DB_PATH)
    if conn is None: 
        return None 
    if not check_database_tables_exist(conn): 
        try: conn.close()
        except: pass
        st.error("Database tables missing or schema incorrect for this app version. Re-run `create_db.py` (simplified version).")
        return None
    return conn

# --- 2. Data Access Functions & Text Processing ---
@st.cache_data
def get_total_entries_count(_conn):
    if not _conn: return 0
    try:
        cursor = _conn.cursor()
        cursor.execute("SELECT COUNT(id) FROM words")
        return cursor.fetchone()[0]
    except: return 0

@st.cache_data
def get_unique_pos_list(_conn):
    if not _conn: return ["Any"]
    try:
        cursor = _conn.cursor()
        cursor.execute("SELECT DISTINCT type FROM definitions WHERE type IS NOT NULL AND type != '' ORDER BY type")
        return ["Any"] + [row[0] for row in cursor.fetchall()]
    except: return ["Any"]

def get_wordnet_pos_for_lemmatizer(treebank_tag):
    global wn 
    if not wn or not wordnet_available: return 'n' 

    if treebank_tag.startswith('J'): return wn.ADJ
    elif treebank_tag.startswith('V'): return wn.VERB
    elif treebank_tag.startswith('N'): return wn.NOUN
    elif treebank_tag.startswith('R'): return wn.ADV
    else: return wn.NOUN 

def preprocess_text_for_keywords(text, stemmer_instance, stop_words_set):
    if not text or not isinstance(text, str): return []
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
    try: tokens = word_tokenize(cleaned_text)
    except LookupError: # If 'punkt' is missing despite initial check
        st.warning("NLTK 'punkt' tokenizer data not found during preprocessing. Results may be affected.")
        return [stemmer_instance.stem(cleaned_text)] # Basic fallback
    except: return [] 
    return sorted(list(set(stemmer_instance.stem(token) for token in tokens
                           if token and token not in stop_words_set and len(token) > 1)))

def preprocess_text_for_tfidf(text, lemmatizer_instance, stop_words_set):
    global wordnet_available, wn 
    if not text or not isinstance(text, str) or not wordnet_available or not wn: return []
    cleaned_text = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
    try: tokens = word_tokenize(cleaned_text)
    except LookupError:
        st.warning("NLTK 'punkt' tokenizer data not found during preprocessing. Results may be affected.")
        return [] # Fallback: cannot proceed without tokenization
    except: return []
    
    try:
        tagged_tokens = nltk.pos_tag(tokens)
    except LookupError:
        st.warning("NLTK 'averaged_perceptron_tagger' data not found. Lemmatization will be less effective (defaulting to nouns).")
        tagged_tokens = [(token, 'NN') for token in tokens] # Fallback: assume all nouns
    except Exception as e:
        tagged_tokens = [(token, 'NN') for token in tokens] 

    lemmatized_tokens = [
        lemmatizer_instance.lemmatize(word, pos=get_wordnet_pos_for_lemmatizer(tag))
        for word, tag in tagged_tokens
        if word not in stop_words_set and len(word) > 1
    ]
    return sorted(list(set(lemmatized_tokens)))

# --- 3. Query Elaboration and Search Function ---

def elaborate_query_with_wordnet(query_text, stemmer_instance, lemmatizer_instance, stop_words_set,
                                 max_senses=1, max_synonyms_per_sense=2):
    global wordnet_available, wn
    
    original_query_stemmed_keywords = preprocess_text_for_keywords(query_text, stemmer_instance, stop_words_set)
    original_query_lemmatized_for_tfidf = preprocess_text_for_tfidf(query_text, lemmatizer_instance, stop_words_set)

    if not query_text.strip() or not wordnet_available or not wn:
        return query_text, set(original_query_stemmed_keywords), " ".join(original_query_lemmatized_for_tfidf)

    lookup_tokens = [word.lower() for word in word_tokenize(query_text)
                     if word.isalpha() and word.lower() not in stop_words_set and len(word) > 1]
    if not lookup_tokens:
         return query_text, set(original_query_stemmed_keywords), " ".join(original_query_lemmatized_for_tfidf)

    elaboration_display_parts = [f"**Original Query:** {query_text}"]
    elaboration_texts_for_tfidf_list = list(original_query_lemmatized_for_tfidf) 
    keywords_for_prefilter_set = set(original_query_stemmed_keywords)

    for token in lookup_tokens:
        try:
            synsets = wn.synsets(token)
            if not synsets: continue

            token_defs_display_list = []
            token_syns_display_list = []

            for i, synset in enumerate(synsets):
                if i < max_senses: 
                    definition = synset.definition()
                    if definition:
                        token_defs_display_list.append(f"- {definition}")
                        elaboration_texts_for_tfidf_list.extend(preprocess_text_for_tfidf(definition, lemmatizer_instance, stop_words_set))

                    for j, lemma in enumerate(synset.lemmas()): 
                        synonym = lemma.name().replace('_', ' ').lower()
                        if synonym != token: 
                            if j < max_synonyms_per_sense: 
                                token_syns_display_list.append(synonym)
                                elaboration_texts_for_tfidf_list.extend(preprocess_text_for_tfidf(synonym, lemmatizer_instance, stop_words_set))
                                keywords_for_prefilter_set.update(preprocess_text_for_keywords(synonym, stemmer_instance, stop_words_set))
            
            if token_defs_display_list:
                elaboration_display_parts.append(f"\n**Regarding '{token}':**\n*Meanings/Context:*\n" + "\n".join(token_defs_display_list))
            if token_syns_display_list:
                elaboration_display_parts.append(f"*Related terms for '{token}':* {', '.join(list(set(token_syns_display_list)))}")
        except LookupError: # Should be caught by initial check, but as a safeguard
            wordnet_available = False # Mark it as unavailable if error occurs during usage
            st.warning(f"WordNet lookup failed for token '{token}'. Elaboration may be incomplete.")
            continue # Skip to next token
        except Exception as e:
            # print(f"Error elaborating token '{token}': {e}") # For debugging
            continue

    display_elaboration_text = "\n\n".join(elaboration_display_parts)
    processed_elaboration_str_for_tfidf = " ".join(sorted(list(set(elaboration_texts_for_tfidf_list))))
    
    return display_elaboration_text, keywords_for_prefilter_set, processed_elaboration_str_for_tfidf

def get_antonyms_for_query(query_text, lemmatizer_instance, stop_words_set):
    global wordnet_available, wn
    if not wordnet_available or not query_text.strip() or not wn: return set()

    processed_query_lemmas = preprocess_text_for_tfidf(query_text, lemmatizer_instance, stop_words_set)
    antonyms_set = set()
    for lemma_q in processed_query_lemmas:
        try:
            for ss in wn.synsets(lemma_q):
                for lm in ss.lemmas():
                    for ant in lm.antonyms():
                        antonyms_set.add(ant.name().replace('_', ' ').lower())
        except LookupError: wordnet_available = False; return set() # Data missing
        except Exception: continue
    final_antonyms_lemmatized = set()
    for ant in antonyms_set:
        final_antonyms_lemmatized.update(preprocess_text_for_tfidf(ant, lemmatizer_instance, stop_words_set))
    return final_antonyms_lemmatized

# Replace your existing nlp_based_search_from_db or adapt it
def sentence_embedding_search_on_the_fly(conn, original_query_text, selected_pos,
                                         stemmer_instance, lemmatizer_instance, stop_words_set,
                                         sentence_model_instance, # New parameter
                                         use_wordnet_elaboration=True): # Keep WordNet elaboration part
    if not conn: st.error("DB connection unavailable."); return []
    if not sentence_model_instance: 
        st.error("Sentence embedding model not loaded. Cannot perform semantic search.")
        return []

    cleaned_original_query = original_query_text.strip()
    if not cleaned_original_query: return []

    # Step 1: Elaborate Query (using your existing WordNet function)
    # elaborate_query_with_wordnet should return:
    # display_elaboration, keywords_for_prefilter, text_to_embed_for_query
    display_elaboration, keywords_for_prefilter, elaborated_text_for_embedding = \
        elaborate_query_with_wordnet(cleaned_original_query, stemmer_instance, lemmatizer_instance, stop_words_set, 
                                     max_senses=1 if use_wordnet_elaboration and wordnet_available else 0,
                                     max_synonyms_per_sense=2 if use_wordnet_elaboration and wordnet_available else 0)

    query_antonyms_lemmatized = set()
    if use_wordnet_elaboration and wordnet_available:
        query_antonyms_lemmatized = get_antonyms_for_query(cleaned_original_query, lemmatizer_instance, stop_words_set)
        if display_elaboration != cleaned_original_query:
            with st.expander("Query Elaboration Context (WordNet)", expanded=False):
                 st.markdown(display_elaboration)
        if query_antonyms_lemmatized:
            st.caption(f"Note: Results with antonyms like '{', '.join(list(query_antonyms_lemmatized)[:5])}...' may be penalized.")

    if not keywords_for_prefilter and not elaborated_text_for_embedding.strip():
        st.info("Could not derive effective keywords or elaboration from query."); return []

    # Generate embedding for the (possibly elaborated) query
    try:
        query_embedding = sentence_model_instance.encode(elaborated_text_for_embedding)
    except Exception as e:
        st.error(f"Error encoding query text ('{elaborated_text_for_embedding[:100]}...'): {e}"); return []

    # Step 2: Keyword Pre-filtering from DB
    # This part remains largely the same, fetching 'definition_entry' for candidates
    is_pos_filter_active = selected_pos and selected_pos.lower() != "any"
    selected_pos_lower = selected_pos.lower() if is_pos_filter_active else ""
    sql_fetch_for_prefilter = "SELECT id as definition_id, word_id, definition_entry, stemmed_words_text FROM definitions WHERE stemmed_words_text IS NOT NULL" # Ensure you fetch definition_entry
    params_prefilter = []
    if is_pos_filter_active: sql_fetch_for_prefilter += " AND LOWER(type) = ?"; params_prefilter.append(selected_pos_lower)

    candidate_definitions_from_db = [] # List of dicts: {'definition_id': ..., 'word_id': ..., 'raw_text': ...}
    cursor = conn.cursor()
    try:
        cursor.execute(sql_fetch_for_prefilter, params_prefilter)
        for row in cursor.fetchall():
            db_stemmed_text = row['stemmed_words_text']
            if not db_stemmed_text: continue
            definition_stemmed_set = set(db_stemmed_text.split(' '))
            if any(keyword in definition_stemmed_set for keyword in keywords_for_prefilter):
                candidate_definitions_from_db.append({'definition_id': row['definition_id'], 
                                                    'word_id': row['word_id'], 
                                                    'raw_text': row['definition_entry']}) # Store raw_text
                if len(candidate_definitions_from_db) >= MAX_PREFILTER_CANDIDATES: break
    except sqlite3.Error as e: st.error(f"Error during keyword pre-filtering: {e}"); return []

    if not candidate_definitions_from_db: st.info("Keyword pre-filter found no candidate definitions."); return []
    # st.write(f"Keyword pre-filter: {len(candidate_definitions_from_db)} candidates. Encoding definitions...")


    # Step 3: On-the-fly Embedding Generation for Candidates & Similarity Calculation
    definition_texts_to_embed = [cand['raw_text'] for cand in candidate_definitions_from_db if cand['raw_text']]
    results_with_scores = []

    if definition_texts_to_embed:
        try:
            # Batch encode all candidate definition texts
            definition_embeddings = sentence_model_instance.encode(definition_texts_to_embed)

            # Calculate cosine similarities between query_embedding and all definition_embeddings
            # query_embedding needs to be reshaped to (1, D) if it's (D,)
            if query_embedding.ndim == 1: query_embedding_reshaped = query_embedding.reshape(1, -1)
            else: query_embedding_reshaped = query_embedding

            similarities = cosine_similarity(query_embedding_reshaped, definition_embeddings).flatten()

            idx = 0 # To map similarities back to candidate_definitions_from_db if some raw_text was empty
            for i, cand_def_info in enumerate(candidate_definitions_from_db):
                if not cand_def_info['raw_text']: continue # Skip if raw_text was empty

                score = float(similarities[idx])
                idx += 1

                # Antonym Penalization (optional)
                if query_antonyms_lemmatized and wordnet_available:
                    definition_lemmas_set = set(preprocess_text_for_tfidf(cand_def_info['raw_text'], lemmatizer_global, english_stopwords_global))
                    if any(antonym in definition_lemmas_set for antonym in query_antonyms_lemmatized):
                        score *= 0.1 

                if score >= SEMANTIC_SIMILARITY_THRESHOLD:
                    results_with_scores.append({**cand_def_info, 'similarity': score})
        except Exception as e:
            st.error(f"Error during sentence embedding of definitions or similarity calculation: {e}"); return []

    results_with_scores.sort(key=lambda x: x['similarity'], reverse=True)
    if not results_with_scores: st.info(f"No definitions passed semantic similarity threshold ({SEMANTIC_SIMILARITY_THRESHOLD})."); return []

    # Step 4: Fetch Full Data & Format Results (this part remains the same)
    # ... (copy the existing Step 4 logic from your script here) ...
    # It will use `results_with_scores` to build `word_max_similarity`, `distinct_word_ids_to_fetch`,
    # and then fetch full word data and definitions to create `final_results_list`.
    word_max_similarity = {}; distinct_word_ids_to_fetch = set()
    for r_def in results_with_scores: 
        w_id = r_def['word_id']; distinct_word_ids_to_fetch.add(w_id)
        if w_id not in word_max_similarity or r_def['similarity'] > word_max_similarity[w_id]: word_max_similarity[w_id] = r_def['similarity']
        if len(distinct_word_ids_to_fetch) >= MAX_FINAL_RESULTS + 20: break 

    if not distinct_word_ids_to_fetch: return []
    final_word_ids_list = sorted(list(distinct_word_ids_to_fetch), key=lambda wid: word_max_similarity.get(wid, 0.0), reverse=True)

    placeholders = ','.join(['?'] * len(final_word_ids_list))
    sql_fetch_words = f"SELECT id, head, entry_text, phone, origin, info FROM words WHERE id IN ({placeholders})"
    sql_fetch_all_definitions = f"SELECT id, word_id, definition_entry, type FROM definitions WHERE word_id IN ({placeholders})"
    words_data_map = {}; definitions_by_word_id_map = {}
    try: 
        cursor.execute(sql_fetch_words, final_word_ids_list); words_data_map = {r['id']: dict(r) for r in cursor.fetchall()}
        cursor.execute(sql_fetch_all_definitions, final_word_ids_list)
        for dr in cursor.fetchall():
            w_id = dr['word_id']
            if w_id not in definitions_by_word_id_map: definitions_by_word_id_map[w_id] = []
            definitions_by_word_id_map[w_id].append({'id': dr['id'], 'entry': dr['definition_entry'], 'type': dr['type']})
    except sqlite3.Error as e: st.error(f"DB error fetching final details: {e}"); return []

    final_results_list = []
    for word_id_val in final_word_ids_list:
        if word_id_val not in words_data_map: continue
        word_info_dict = words_data_map[word_id_val]
        final_results_list.append({
            'id': word_info_dict['id'], 'entry': word_info_dict['entry_text'], 'head': word_info_dict['head'], 
            'phone': word_info_dict['phone'], 'origin': word_info_dict['origin'], 'info': word_info_dict['info'],
            'defs': definitions_by_word_id_map.get(word_id_val, []), 'max_similarity': word_max_similarity.get(word_id_val, 0.0) 
        })
        if len(final_results_list) >= MAX_FINAL_RESULTS: break
    return final_results_list


# --- 4. Streamlit User Interface ---
# st.set_page_config(page_title="Alar.ink Definition Search", layout="wide")

# Using your provided st.title and st.markdown
st.title("📖 [alar.ink](https://alar.ink/) Corpus English Definition Search")
st.markdown("""
Search within [alar.ink](https://alar.ink/)'s English definitions for a matching word in Kannada and filter by part of speech (type).
Consider it a makeshift English-Kannada lookup as V. Krishna and Kailash Nadh ಅವರು work on the real thing. 
Many thanks to them both for their hard work to make alar.ink possible - V. Krishna ಅವರೇ has worked on this for 50+ years!

ವಿ. ಕೃಷ್ಣ ಮತ್ತು ಕೈಲಾಶ್ ನಾದ್ ಅವರೇ:
ನಿಮ್ಮ ಪರಿಶ್ರಮಕ್ಕೆ ಹೃತ್ಪೂರ್ವಕ ಧನ್ಯವಾದಗಳು. ಈ ಅದ್ಭುತ ನಿಘಂಟು ನನಗೆ ಕನ್ನಡವನ್ನು ಕಲಿಯಲು ಅಪಾರ ನೆರವು ನೀಡಿದೆ. 
ವಿ. ಕೃಷ್ಣ ಅವರೇ, ನಿಮ್ಮ ಐವತ್ತು ವರ್ಷದ ಪ್ರಯತ್ನ ಬಗ್ಗೆ ಓದುವಾಗ ನಿಮ್ಮ ನಿದರ್ಶನವನ್ನು ಅನುಸರಿಸಲು ನನಗೆ ಪ್ರೇರಣೆಯಾಗಿದೆ. 
""")

if not sentence_transformers_available or sentence_model_global is None:
    st.sidebar.error("Sentence Transformer model not available. Semantic search functionality is disabled.")
    # You might want to fall back to a simpler search or just show an error if this is the primary search method.


if not wordnet_available: # Check the flag set by initial NLTK setup
    st.sidebar.warning("WordNet data is not fully available. Query elaboration will be basic or disabled.")

db_connection = initialize_database_connection()

if db_connection is None:
    st.warning("Application cannot proceed: Database not ready. Please run create_db.py."); st.stop()

total_entries_in_db = get_total_entries_count(db_connection)
unique_pos_list_from_db = get_unique_pos_list(db_connection)

if total_entries_in_db == 0:
    st.error(f"The database '{DB_PATH}' is empty. Run create_db.py."); st.stop()

col1, col2 = st.columns([3, 1]) 
with col1:
    search_definition_term_input = st.text_input("Describe the definition you're looking for:", placeholder="e.g., a feeling of joy, instrument for writing")
with col2:
    selected_part_of_speech_filter = st.selectbox("Part of Speech (optional):", unique_pos_list_from_db)

use_wordnet_elaboration_checkbox = st.checkbox("Elaborate query with WordNet (more context)", value=True, disabled=(not wordnet_available))
if not wordnet_available and use_wordnet_elaboration_checkbox:
    st.caption("WordNet elaboration disabled as NLTK data is unavailable.")

should_use_wordnet_for_elaboration = use_wordnet_elaboration_checkbox and wordnet_available # <<< CORRECT LOGIC

if search_definition_term_input and search_definition_term_input.strip():
    st.markdown("---") 
    with st.spinner("Elaborating query and searching... This may take a moment."):
        search_results_list = sentence_embedding_search_on_the_fly( # Call the new function
            db_connection, 
            search_definition_term_input, 
            selected_part_of_speech_filter,
            stemmer_global, 
            lemmatizer_global, # Pass lemmatizer
            english_stopwords_global, 
            sentence_model_global, # Pass the loaded sentence model
            use_wordnet_elaboration = should_use_wordnet_for_elaboration
            # Removed LLM params
        )

    total_found_count = len(search_results_list)

    if search_results_list:
        subheader_message_parts = [f"definitions related to your query \"{search_definition_term_input.strip()}\""]
        if use_wordnet_elaboration_checkbox and wordnet_available:
             subheader_message_parts.append("(WordNet elaborated)")
        if selected_part_of_speech_filter and selected_part_of_speech_filter != "Any":
            subheader_message_parts.append(f"with part of speech \"{selected_part_of_speech_filter.capitalize()}\"")
        
        criteria_message = " ".join(subheader_message_parts)
        st.subheader(f"Found {total_found_count} relevant entr{'y' if total_found_count == 1 else 'ies'} for {criteria_message}:")
        
        if total_found_count >= MAX_FINAL_RESULTS: 
            st.info(f"Displaying the top {MAX_FINAL_RESULTS} matching entries.")

        for i, entry_data_dict in enumerate(search_results_list):
            display_word = entry_data_dict.get('entry', "Unknown Entry")
            expander_pos_display = "N/A"; current_definitions_list = entry_data_dict.get("defs", [])
            if current_definitions_list:
                temp_pos_list = [d.get("type","").strip().lower() for d in current_definitions_list if d.get("type")]
                if selected_part_of_speech_filter and selected_part_of_speech_filter.lower() != "any":
                    if selected_part_of_speech_filter.lower() in temp_pos_list: expander_pos_display = selected_part_of_speech_filter.capitalize()
                    elif temp_pos_list: expander_pos_display = temp_pos_list[0].capitalize()
                elif temp_pos_list: expander_pos_display = temp_pos_list[0].capitalize()
            
            similarity_score_info = f"(Relevance: {entry_data_dict.get('max_similarity', 0.0):.2f})" if entry_data_dict.get('max_similarity') is not None else ""

            with st.expander(f"{display_word} ({expander_pos_display}) {similarity_score_info}", expanded=(i < 2)):
                if entry_data_dict.get('phone'): st.markdown(f"**Phonetic:** {entry_data_dict.get('phone')}")
                if entry_data_dict.get('origin'): st.markdown(f"**Origin:** {entry_data_dict.get('origin')}")
                if entry_data_dict.get('info'): st.markdown(f"**Info:** {entry_data_dict.get('info')}")
                st.markdown("**Definitions:**")
                if current_definitions_list:
                    for def_idx, definition_dict in enumerate(current_definitions_list):
                        st.markdown(f"- **{definition_dict.get('entry', 'N/A')}** (*{definition_dict.get('type', 'N/A')}*)")
                        if def_idx < len(current_definitions_list) - 1: st.markdown("---") 
                else: st.write("No definitions listed.")
    else: 
        st.info(f"No entries found for \"{search_definition_term_input.strip()}\". Try rephrasing or different keywords.")
else:
    st.info("Enter a query to find definitions.")

# Using your provided sidebar text
st.sidebar.header("About")
st.sidebar.info(
    "This application allows you to search within dictionary definitions "
    "and filter by part of speech (type).\n\n"
    "It loads data from an the alar.ink corpus created by V. Krishna and Kailash Nadh."
)
st.sidebar.markdown("---")
if db_connection:
    st.sidebar.markdown(f"Total entries loaded from alar.ink: **{get_total_entries_count(db_connection)}**")