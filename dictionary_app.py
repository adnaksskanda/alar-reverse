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

from llama_cpp import Llama
llama_cpp_available = True

# --- Attempt to load NLTK resources and set global variables ---
wn = None
english_stopwords_global = set()
wordnet_available = False

# --- Sentence Transformer Model Loading ---
# Placed after global configurations
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2' # A good default
SEMANTIC_SIMILARITY_THRESHOLD = 0.3      # Adjust for sentence embeddings

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

MODEL_REPO_ID = "MoMonir/Phi-3-mini-128k-instruct-GGUF"
MODEL_FILENAME = "phi-3-mini-128k-instruct.Q4_K_M.gguf"
N_GPU_LAYERS_HEROKU = 0 # Important: Heroku free/hobby dynos don't have GPUs. Set to 0.

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
def load_local_phi3_model_from_hub():
    global llama_cpp_available # Assuming this flag is set based on import
    if not llama_cpp_available:
        st.sidebar.warning("`llama-cpp-python` library not installed. LLM features disabled.")
        return None
    try:
        st.sidebar.info(f"Downloading/loading LLM: {MODEL_REPO_ID}/{MODEL_FILENAME}...")
        # This will download from Hugging Face Hub to a local cache directory
        # if not already present in the cache.
        llm = Llama.from_pretrained(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            n_ctx=500,       # Or your desired context size
            n_gpu_layers=N_GPU_LAYERS_HEROKU, # Must be 0 for Heroku free/hobby dynos
            verbose=False    
        )
        st.sidebar.success("Local LLM loaded successfully.")
        return llm
    except Exception as e:
        st.sidebar.error(f"Error loading LLM from Hub: {e}")
        st.sidebar.caption("This might be due to download issues, model compatibility, or llama-cpp-python setup.")
        return None

local_llm_instance = load_local_phi3_model_from_hub()


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

sentence_model_global = load_sentence_embedding_model(SENTENCE_MODEL_NAME)

db_connection = initialize_database_connection() # Your main dictionary DB
freq_db_connection = get_frequency_db_connection(KANNADA_FREQUENCY_DB_PATH) # <<< ADD THIS CALL

# Update UI messages based on freq_db_connection status
if db_connection is None: 
    st.warning("Application cannot proceed: Main Database not ready."); st.stop()
if freq_db_connection is None: # This is a global check now
    st.sidebar.warning(f"Frequency DB ('{KANNADA_FREQUENCY_DB_PATH}') not loaded. Commonness scores will not be used in ranking.")



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

def elaborate_query_with_local_llm(query_text, llm_instance, 
                                   stemmer_instance, lemmatizer_instance, stop_words_set):
    """Elaborates query using a local LLM instance (e.g., Phi-3 GGUF)."""
    fallback_display = f"**Original Query:** {query_text}\n\n*(Local LLM elaboration was skipped or failed.)*"
    # Prepare fallback outputs based on original query
    original_q_stemmed_keywords = preprocess_text_for_keywords(query_text, stemmer_instance, stop_words_set)
    original_q_lemmatized_str_for_tfidf = " ".join(preprocess_text_for_tfidf(query_text, lemmatizer_instance, stop_words_set))
    
    if not llm_instance or not query_text.strip():
        return fallback_display, set(original_q_stemmed_keywords), original_q_lemmatized_str_for_tfidf

    system_prompt = "You are an expert lexicographer. Your task is to provide a concise, dictionary-like elaboration or a contextual definition for the user's search query. Focus on the core meaning, common contexts, or distinguishing features that would help someone understand what kind of dictionary definition they are looking for. Output a single, informative paragraph of 2-3 sentences."
    user_prompt = f"My dictionary search query is: \"{query_text}\". Please provide the elaboration."
    
    llm_elaboration_text = "" # Initialize
    try:
        # Using create_chat_completion for instruct models like Phi-3
        response = llm_instance.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=50,  # Max length of the generated elaboration
            temperature=0.3, # Lower for more factual/focused output
            stop=["<|end|>", "<|user|>", "<|system|>", "\n\n"] # Common stop tokens for Phi-3
        )
        if response and response['choices'] and response['choices'][0]['message']['content']:
            llm_elaboration_text = response['choices'][0]['message']['content'].strip()
        else:
            # If LLM response is empty or unexpected, use original query for further processing
            llm_elaboration_text = query_text 
            
    except Exception as e:
        st.warning(f"Local LLM query elaboration failed: {e}. Using basic query processing based on original query.")
        llm_elaboration_text = query_text # Fallback to original query text if LLM call fails
    
    # If LLM failed or returned empty or just the query, use the fallback outputs
    if not llm_elaboration_text.strip() or llm_elaboration_text == query_text:
        return fallback_display, set(original_q_stemmed_keywords), original_q_lemmatized_str_for_tfidf
    else:
        # Successfully got an elaboration from LLM
        display_elaboration = f"**Original Query:** {query_text}\n\n**LLM Elaboration:**\n{llm_elaboration_text}"
        
        # Process the LLM elaboration for keywords (stemmed)
        llm_elaboration_stemmed_keywords = preprocess_text_for_keywords(llm_elaboration_text, stemmer_instance, stop_words_set)
        # Combine with original query's stemmed keywords for a richer pre-filter set
        keywords_for_prefilter = set(original_q_stemmed_keywords).union(llm_elaboration_stemmed_keywords)
        
        # Process the LLM elaboration for TF-IDF (lemmatized)
        processed_elaboration_tokens_for_tfidf = preprocess_text_for_tfidf(llm_elaboration_text, lemmatizer_instance, stop_words_set)
        # Combine with original query's lemmatized tokens for TF-IDF query string
        combined_tfidf_tokens = set(original_q_lemmatized_for_tfidf.split()).union(set(processed_elaboration_tokens_for_tfidf))
        processed_elaboration_str_for_tfidf = " ".join(sorted(list(combined_tfidf_tokens)))


        return display_elaboration, keywords_for_prefilter, processed_elaboration_str_for_tfidf

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
# ... (imports and other functions like get_db_connection, preprocess_text, elaborate_query, get_antonyms etc. are assumed to be the same) ...

# Inside your Streamlit app script:

def nlp_based_search_from_db(conn, freq_conn, original_query_text, selected_pos, 
                             stemmer_instance, lemmatizer_instance, stop_words_set, 
                             elaboration_choice="WordNet", # New: "WordNet", "Local LLM", or "None"
                             llm_instance_passed=None):
    if not conn: st.error("DB connection unavailable."); return []
    cleaned_original_query = original_query_text.strip()
    if not cleaned_original_query: return []

    # Step 1: Elaborate Query based on choice
    if elaboration_choice == "Local LLM" and llm_instance_passed:
        display_elaboration, keywords_for_prefilter, processed_elaboration_str_for_tfidf = \
            elaborate_query_with_local_llm(cleaned_original_query, llm_instance_passed, 
                                           stemmer_instance, lemmatizer_instance, stop_words_set)
    elif elaboration_choice == "WordNet" and wordnet_available:
        display_elaboration, keywords_for_prefilter, processed_elaboration_str_for_tfidf = \
            elaborate_query_with_wordnet(cleaned_original_query, stemmer_instance, lemmatizer_instance, stop_words_set)
    else: # No elaboration or fallback
        display_elaboration = f"**Original Query:** {cleaned_original_query}"
        keywords_for_prefilter = set(preprocess_text_for_keywords(cleaned_original_query, stemmer_instance, stop_words_set))
        processed_elaboration_str_for_tfidf = " ".join(preprocess_text_for_tfidf(cleaned_original_query, lemmatizer_instance, stop_words_set))

    query_antonyms_lemmatized = set()
    if wordnet_available: # Antonyms still rely on WordNet
        query_antonyms_lemmatized = get_antonyms_for_query(cleaned_original_query, lemmatizer_instance, stop_words_set)
    
    # Display elaboration context if it happened and is different from original
    if display_elaboration != f"**Original Query:** {cleaned_original_query}" and display_elaboration != cleaned_original_query : 
            with st.expander("Query Elaboration Context", expanded=False): st.markdown(display_elaboration)
    if query_antonyms_lemmatized:
        st.caption(f"Note: Results with terms like '{', '.join(list(query_antonyms_lemmatized)[:5])}...' may be penalized.")

    if not keywords_for_prefilter and not processed_elaboration_str_for_tfidf.strip():
        st.info("Could not derive effective keywords or elaboration from query for search."); return []

    # ... [Rest of the nlp_based_search_from_db function (Steps 2, 3, 4 for pre-filtering, TF-IDF, result formatting)
    #      remains IDENTICAL to your script. It uses `keywords_for_prefilter` and `processed_elaboration_str_for_tfidf` correctly.] ...
    is_pos_filter_active = selected_pos and selected_pos.lower() != "any"
    selected_pos_lower = selected_pos.lower() if is_pos_filter_active else ""
    sql_fetch_for_prefilter = "SELECT id as definition_id, word_id, definition_entry, stemmed_words_text FROM definitions WHERE stemmed_words_text IS NOT NULL"
    params_prefilter = []
    if is_pos_filter_active: sql_fetch_for_prefilter += " AND LOWER(type) = ?"; params_prefilter.append(selected_pos_lower)
    
    candidate_definitions_for_tfidf = []
    cursor = conn.cursor()
    try:
        cursor.execute(sql_fetch_for_prefilter, params_prefilter)
        for row in cursor.fetchall():
            db_stemmed_text = row['stemmed_words_text']
            if not db_stemmed_text: continue 
            definition_stemmed_set = set(db_stemmed_text.split(' '))
            if any(keyword in definition_stemmed_set for keyword in keywords_for_prefilter):
                candidate_definitions_for_tfidf.append({'definition_id': row['definition_id'], 'word_id': row['word_id'], 'raw_text': row['definition_entry']})
                if len(candidate_definitions_for_tfidf) >= MAX_PREFILTER_CANDIDATES: break
    except sqlite3.Error as e: st.error(f"Error during keyword pre-filtering: {e}"); return []

    if not candidate_definitions_for_tfidf: st.info("Keyword pre-filter found no initial candidate definitions."); return []
    
    definition_docs_lemmatized_for_tfidf = [" ".join(preprocess_text_for_tfidf(cd['raw_text'], lemmatizer_instance, stop_words_set)) for cd in candidate_definitions_for_tfidf]
    valid_candidates_after_lemmatization = []
    final_def_docs_for_tfidf = []
    for i, doc_text in enumerate(definition_docs_lemmatized_for_tfidf):
        if doc_text.strip(): valid_candidates_after_lemmatization.append(candidate_definitions_for_tfidf[i]); final_def_docs_for_tfidf.append(doc_text)
    
    if not final_def_docs_for_tfidf or not processed_elaboration_str_for_tfidf.strip(): st.info("No processable text for TF-IDF."); return []

    all_texts_for_vectorization = [processed_elaboration_str_for_tfidf] + final_def_docs_for_tfidf

    
    results_with_scores = [] # This will hold items with 'relevance_score'
    try:
        if len(all_texts_for_vectorization) < 2: st.info("Not enough content for TF-IDF."); return []
        vectorizer = TfidfVectorizer(ngram_range=(1,2)) # As per previous discussion
        tfidf_matrix = vectorizer.fit_transform(all_texts_for_vectorization)
        if tfidf_matrix.shape[1] == 0: st.info("TF-IDF vocabulary empty."); return []
        query_tfidf_vector = tfidf_matrix[0]; definition_tfidf_vectors = tfidf_matrix[1:]

        if definition_tfidf_vectors.shape[0] > 0:
            cosine_similarities = cosine_similarity(query_tfidf_vector, definition_tfidf_vectors).flatten()
            for i, cand_def_info in enumerate(valid_candidates_after_lemmatization): # Use valid_candidates_after_lemmatization
                relevance_score = float(cosine_similarities[i])
                if relevance_score >= SIMILARITY_THRESHOLD_TFIDF: # Apply TF-IDF threshold here
                    results_with_scores.append({
                        **cand_def_info, 
                        'relevance_score': relevance_score # Store raw TF-IDF relevance
                    })

    except ValueError as ve: st.warning(f"TF-IDF Error: {ve}."); return []
    except Exception as e: st.error(f"Unexpected error during TF-IDF: {e}"); return []

    if not results_with_scores: 
        st.info(f"No definitions passed TF-IDF similarity threshold ({SIMILARITY_THRESHOLD_TFIDF}).")
        return []

    # --- NEW: Integrate Commonness Score (after initial TF-IDF filtering) ---
    kannada_headwords_map = {} # word_id -> kannada_word
    if freq_conn:
        candidate_word_ids_for_freq = list(set(r['word_id'] for r in results_with_scores))
        if candidate_word_ids_for_freq:
            placeholders_freq = ','.join(['?'] * len(candidate_word_ids_for_freq))
            sql_words = f"SELECT id, entry_text FROM words WHERE id IN ({placeholders_freq})"
            try:
                # Ensure cursor is from the main DB connection (conn)
                main_db_cursor = conn.cursor() 
                main_db_cursor.execute(sql_words, candidate_word_ids_for_freq)
                for row_word in main_db_cursor.fetchall():
                    kannada_headwords_map[row_word['id']] = row_word['entry_text']
            except sqlite3.Error as e:
                st.warning(f"Could not fetch Kannada headwords for frequency: {e}")

        raw_frequencies_in_batch = []
        for res_item in results_with_scores:
            kannada_word = kannada_headwords_map.get(res_item['word_id'])
            raw_freq = get_kannada_word_frequency(freq_conn, kannada_word) if kannada_word else 0
            res_item['raw_frequency'] = raw_freq
            if raw_freq > 0:
                raw_frequencies_in_batch.append(raw_freq)

        norm_commonness_scores_map = {} 
        if raw_frequencies_in_batch: 
            log_frequencies_for_scaling = np.log1p(np.array(raw_frequencies_in_batch, dtype=float))
            if len(log_frequencies_for_scaling) > 0:
                min_log_freq = np.min(log_frequencies_for_scaling)
                max_log_freq = np.max(log_frequencies_for_scaling)
                for res_item in results_with_scores:
                    if res_item['raw_frequency'] > 0:
                        log_f = np.log1p(res_item['raw_frequency'])
                        if (max_log_freq - min_log_freq) > 1e-9:
                            norm_commonness_scores_map[res_item['word_id']] = (log_f - min_log_freq) / (max_log_freq - min_log_freq)
                        elif len(log_frequencies_for_scaling) > 0 : 
                            norm_commonness_scores_map[res_item['word_id']] = 1.0 
                        else: norm_commonness_scores_map[res_item['word_id']] = 0.0
                    else: norm_commonness_scores_map[res_item['word_id']] = 0.0
            else: # All raw frequencies were 0, so all norm scores are 0
                for res_item in results_with_scores: norm_commonness_scores_map[res_item['word_id']] = 0.0
        else: 
            for res_item in results_with_scores: norm_commonness_scores_map[res_item['word_id']] = 0.0

    # Calculate final_score for each item in results_with_scores
    if query_antonyms_lemmatized and wordnet_available:
        st.caption(f"Note: Results may be penalized for antonyms like '{', '.join(list(query_antonyms_lemmatized)[:5])}...'.")

    for res_item in results_with_scores:
        penalized_relevance = res_item['relevance_score'] # Start with TF-IDF score

        if query_antonyms_lemmatized and wordnet_available:
            definition_lemmas_set = set(preprocess_text_for_tfidf(res_item['raw_text'], lemmatizer_instance, stop_words_set))
            if any(antonym in definition_lemmas_set for antonym in query_antonyms_lemmatized):
                penalized_relevance *= 0.1 # Apply penalty

        res_item['penalized_relevance_score'] = penalized_relevance # Store this component

        # Get normalized commonness for this item
        current_normalized_commonness = 0.0
        if freq_conn and res_item.get('raw_frequency', 0) > 0:
            current_normalized_commonness = norm_commonness_scores_map.get(res_item['word_id'], 0.0)

        res_item['commonness_score_normalized_val'] = current_normalized_commonness # Store for display

        # Combine scores
        if freq_conn and res_item.get('raw_frequency', 0) > 0:
            res_item['final_score'] = (RELEVANCE_WEIGHT * penalized_relevance) + \
                                    (COMMONNESS_WEIGHT * current_normalized_commonness)
        else: # Word not found in freq DB or freq_db not connected
            res_item['final_score'] = penalized_relevance

    results_with_scores.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
    # --- End of Commonness Score Integration ---

    # Step 4: Fetch Full Data & Format Results (This part now uses the re-sorted results_with_scores)
    # word_max_similarity should now be word_max_final_score and store the 'final_score' and component scores
    word_top_scores_details = {} 
    distinct_word_ids_to_fetch = set()

    for r_def in results_with_scores: # results_with_scores is already sorted by final_score
        w_id = r_def['word_id']
        if w_id not in distinct_word_ids_to_fetch: # Ensure we only process each word once for its top score
            distinct_word_ids_to_fetch.add(w_id)
            word_top_scores_details[w_id] = {
                'final_score': r_def.get('final_score', 0.0),
                'relevance': r_def.get('penalized_relevance_score', r_def.get('relevance_score', 0.0)),
                'commonness': r_def.get('commonness_score_normalized_val', 0.0)
            }
        if len(distinct_word_ids_to_fetch) >= MAX_FINAL_RESULTS + 20: # Consider enough words for the final cut
            break 

    if not distinct_word_ids_to_fetch: return []

    # Sort the word_ids by the final_score we stored for them
    final_word_ids_list_sorted = sorted(
        list(distinct_word_ids_to_fetch), 
        key=lambda wid: word_top_scores_details.get(wid, {}).get('final_score', 0.0), 
        reverse=True
    )

    # Fetch full details for these words
    placeholders = ','.join(['?'] * len(final_word_ids_list_sorted))
    sql_fetch_words = f"SELECT id, head, entry_text, phone, origin, info FROM words WHERE id IN ({placeholders})"
    sql_fetch_all_definitions = f"SELECT id, word_id, definition_entry, type FROM definitions WHERE word_id IN ({placeholders})"
    words_data_map = {}; definitions_by_word_id_map = {}
    try: 
        # Re-ensure cursor is valid from the main connection `conn`
        cursor = conn.cursor() 
        cursor.execute(sql_fetch_words, final_word_ids_list_sorted); words_data_map = {r['id']: dict(r) for r in cursor.fetchall()}
        cursor.execute(sql_fetch_all_definitions, final_word_ids_list_sorted)
        for dr in cursor.fetchall():
            w_id = dr['word_id']
            if w_id not in definitions_by_word_id_map: definitions_by_word_id_map[w_id] = []
            definitions_by_word_id_map[w_id].append({'id': dr['id'], 'entry': dr['definition_entry'], 'type': dr['type']})
    except sqlite3.Error as e: st.error(f"DB error fetching final details: {e}"); return []

    final_results_list = []
    for word_id_val in final_word_ids_list_sorted: 
        if word_id_val not in words_data_map: continue
        word_info_dict = words_data_map[word_id_val]
        scores = word_top_scores_details.get(word_id_val, {})

        final_results_list.append({
            'id': word_info_dict['id'], 'entry': word_info_dict['entry_text'], 
            'head': word_info_dict['head'], 'phone': word_info_dict['phone'],
            'origin': word_info_dict['origin'], 'info': word_info_dict['info'],
            'defs': definitions_by_word_id_map.get(word_id_val, []), 
            'final_score': scores.get('final_score', 0.0), # Overall combined score
            'relevance_score_val': scores.get('relevance', 0.0), # Penalized Relevance
            'commonness_score_val': scores.get('commonness', 0.0) # Normalized Commonness
        })
        if len(final_results_list) >= MAX_FINAL_RESULTS: break # Apply final display limit

    return final_results_list

# --- 4. Streamlit User Interface ---
# Using your provided st.title and st.markdown

# --- 4. Streamlit User Interface ---
# Assuming all necessary functions (initialize_database_connection, get_total_entries_count,
# get_unique_pos_list, nlp_based_search_from_db, etc.) and global variables 
# (db_connection, freq_db_connection, local_llm_instance, wordnet_available, 
#  stemmer_global, lemmatizer_global, english_stopwords_global)
# are defined and initialized correctly before this section, as per our previous discussions.

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

# --- Initialize Connections and Check Resources ---
# These initializations happen globally now, before this UI section typically.
# For clarity, this is where you ensure they are ready:

# db_connection = initialize_database_connection() # Main dictionary DB
# freq_db_connection = get_frequency_db_connection(KANNADA_FREQUENCY_DB_PATH) # Kannada Freq DB
# local_llm_instance is also loaded globally via @st.cache_resource
# wordnet_available is set globally after NLTK checks

# --- UI Messages for Resource Status ---
if not wordnet_available: 
    st.sidebar.warning("WordNet NLTK data not fully available. WordNet-based query elaboration will be basic or disabled.")

# Check for local LLM components
if not llama_cpp_available: # This global flag is set based on `from llama_cpp import Llama`
    st.sidebar.warning("`llama-cpp-python` library not installed. Local LLM features will be disabled.")
elif local_llm_instance is None and llama_cpp_available: 
     # This implies library is there, but model file itself failed to load via load_local_phi3_model()
     st.sidebar.error(f"Local LLM model specified ('{LOCAL_LLM_MODEL_PATH}') failed to load or not found. LLM elaboration disabled.")

if freq_db_connection is None:
    st.sidebar.warning(f"Frequency DB ('{KANNADA_FREQUENCY_DB_PATH}') not loaded. Commonness scores will not be used in ranking.")


# --- Main App Logic ---
if db_connection is None:
    st.error("Main Database connection failed. Application cannot proceed. Please check setup and console logs.")
    st.stop()

total_entries_in_db = get_total_entries_count(db_connection)
unique_pos_list_from_db = get_unique_pos_list(db_connection)

if total_entries_in_db == 0:
    st.error(f"The main database '{DB_PATH}' is empty or could not be read. Please run `create_db.py` to populate it.")
    st.stop()

col1, col2 = st.columns([3, 1]) 
with col1:
    search_definition_term_input = st.text_input("Describe the definition you're looking for:", placeholder="e.g., a feeling of joy, instrument for writing")
with col2:
    selected_part_of_speech_filter = st.selectbox("Part of Speech (optional):", unique_pos_list_from_db)

# --- Elaboration Method Selection ---
elaboration_method_options = ["None (Raw Query)"] 
elaboration_method_default_idx = 0

if wordnet_available:
    elaboration_method_options.append("WordNet (fast, basic)")
    if elaboration_method_default_idx == 0 and "None" in elaboration_method_options[0]: # Prefer WordNet over None
        elaboration_method_default_idx = len(elaboration_method_options) -1

if local_llm_instance: # Only add LLM option if the model instance loaded successfully
    elaboration_method_options.append("Local LLM (Phi-3 mini, richer context)")
    elaboration_method_default_idx = len(elaboration_method_options) -1 # Prefer LLM if available

elaboration_choice = st.radio(
    "Query Elaboration Method:",
    options=elaboration_method_options,
    index=elaboration_method_default_idx, 
    horizontal=True,
    disabled=(len(elaboration_method_options) <= 1 and elaboration_method_options[0] == "None (Raw Query)")
)

if elaboration_choice == "None (Raw Query)" and (wordnet_available or local_llm_instance):
    st.caption("Using raw query for search. You can select WordNet or LLM for richer context if available.")
elif not wordnet_available and not local_llm_instance and elaboration_choice != "None (Raw Query)":
    st.caption("Chosen elaboration method unavailable. Search will use raw query.")


# --- Perform Search and Display Results ---
if search_definition_term_input and search_definition_term_input.strip():
    st.markdown("---") 
    
    # Determine which elaboration method is effectively active
    active_elaboration_method = "None"
    if elaboration_choice == "Local LLM (Phi-3 mini, richer context)" and local_llm_instance:
        active_elaboration_method = "LLM"
    elif elaboration_choice == "WordNet (fast, basic)" and wordnet_available:
        active_elaboration_method = "WordNet"

    spinner_message = f"Searching (using {active_elaboration_method} elaboration)..."
    if active_elaboration_method == "LLM":
        spinner_message = "Elaborating with Local LLM and searching (this can be slow)..."
    
    with st.spinner(spinner_message):
        search_results_list = nlp_based_search_from_db( # This is your TF-IDF based search function
            db_connection, 
            freq_db_connection, # Pass the frequency DB connection
            search_definition_term_input, 
            selected_part_of_speech_filter,
            stemmer_global, 
            lemmatizer_global,
            english_stopwords_global, 
            elaboration_choice=elaboration_choice, # Pass the string choice
            llm_instance_passed=local_llm_instance   # Pass the LLM instance
        )
    total_found_count = len(search_results_list)

    if search_results_list:
        subheader_message_parts = [f"definitions related to \"{search_definition_term_input.strip()}\""]
        if active_elaboration_method == "LLM":
            subheader_message_parts.append("(LLM elaborated)")
        elif active_elaboration_method == "WordNet":
            subheader_message_parts.append("(WordNet elaborated)")
        
        if selected_part_of_speech_filter and selected_part_of_speech_filter != "Any":
            subheader_message_parts.append(f"with Part of Speech \"{selected_part_of_speech_filter.capitalize()}\"")
        
        criteria_message = " ".join(subheader_message_parts)
        st.subheader(f"Found {total_found_count} relevant entr{'y' if total_found_count == 1 else 'ies'} for {criteria_message}:")
        
        if total_found_count >= MAX_FINAL_RESULTS: 
            st.info(f"Displaying the top {MAX_FINAL_RESULTS} matching entries.")

        for i, entry_data_dict in enumerate(search_results_list):
            display_word = entry_data_dict.get('entry', "Unknown Entry")
            expander_pos_display = "N/A"
            current_definitions_list = entry_data_dict.get("defs", [])
            if current_definitions_list: # Logic to determine expander_pos_display
                temp_pos_list = [d.get("type","").strip().lower() for d in current_definitions_list if d.get("type")]
                if selected_part_of_speech_filter and selected_part_of_speech_filter.lower() != "any":
                    if selected_part_of_speech_filter.lower() in temp_pos_list: 
                        expander_pos_display = selected_part_of_speech_filter.capitalize()
                    elif temp_pos_list: 
                        expander_pos_display = temp_pos_list[0].capitalize()
                elif temp_pos_list: 
                     expander_pos_display = temp_pos_list[0].capitalize()
            
            # --- Updated Score Display ---
            final_s = entry_data_dict.get('final_score', 0.0)
            relevance_s = entry_data_dict.get('relevance_score_val', 0.0) # This should be the penalized TF-IDF score
            commonness_s = entry_data_dict.get('commonness_score_val', 0.0) # Normalized 0-1

            score_strings = [f"Overall: {final_s:.2f}"]
            # Add relevance if it's meaningful to distinguish from final score
            if abs(final_s - relevance_s) > 0.001 or not freq_db_connection or entry_data_dict.get('raw_frequency', 0) == 0:
                score_strings.append(f"Rel: {relevance_s:.2f}")
            
            if freq_db_connection and entry_data_dict.get('raw_frequency', -1) != -1 : # Check if commonness was attempted
                # Display commonness if it was factored in (raw_frequency > 0) or if freq_db was available
                # commonness_s will be 0 if word not found or freq_db was off.
                # We only want to show "Common: 0.00" if the word was looked up and found to be 0 or least common.
                # If raw_frequency exists and freq_db_connection is on, it means commonness was processed.
                if 'raw_frequency' in entry_data_dict: # Indicates commonness processing occurred for this item
                     score_strings.append(f"Common: {commonness_s:.2f}")
            
            similarity_score_info = f"({', '.join(score_strings)})"
            # --- End of Updated Score Display ---

            with st.expander(f"{display_word} ({expander_pos_display}) {similarity_score_info}", expanded=(i < 2)): # Expand top 2 results
                if entry_data_dict.get('phone'): st.markdown(f"**Phonetic:** {entry_data_dict.get('phone')}")
                if entry_data_dict.get('origin'): st.markdown(f"**Origin:** {entry_data_dict.get('origin')}")
                if entry_data_dict.get('info'): st.markdown(f"**Info:** {entry_data_dict.get('info')}")
                st.markdown("**Definitions:**")
                if current_definitions_list:
                    for def_idx, definition_dict in enumerate(current_definitions_list):
                        st.markdown(f"- **{definition_dict.get('entry', 'N/A')}** (*{definition_dict.get('type', 'N/A')}*)")
                        if def_idx < len(current_definitions_list) - 1: st.markdown("---") 
                else: st.write("No definitions listed for this entry.")
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
if db_connection: # Check if db_connection is not None
    st.sidebar.markdown(f"Total entries loaded from alar.ink: **{get_total_entries_count(db_connection)}**")
