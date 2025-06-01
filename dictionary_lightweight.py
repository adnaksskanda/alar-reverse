import streamlit as st
import sqlite3
import os
import re
import numpy as np

# NLTK imports
import nltk
nltk.data.path.append(os.path.join(os.path.dirname(__file__), 'nltk_data'))
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Scikit-learn import for cosine_similarity (TfidfVectorizer is removed)
from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import TfidfVectorizer # REMOVED

# Sentence Transformer and Llama CPP imports
try:
    from sentence_transformers import SentenceTransformer
    sentence_transformers_available = True
except ImportError:
    sentence_transformers_available = False
    print("WARNING: sentence-transformers library not found. Sentence embedding features disabled.")


# --- Global NLTK Components & Flags ---
wn = None
english_stopwords_global = set()
wordnet_available = False

try:
    from nltk.corpus import wordnet as wn_imported
    from nltk.corpus import stopwords as stopwords_imported
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('taggers/averaged_perceptron_tagger')
    nltk.data.find('corpora/wordnet'); nltk.data.find('corpora/omw-1.4'); nltk.data.find('corpora/stopwords')
    wn = wn_imported
    english_stopwords_global = set(stopwords_imported.words('english'))
    if wn:
        wn.synsets("test")
        wordnet_available = True
        print("NLTK resources appear available.")
    else:
        print("nltk.corpus.wordnet could not be imported/assigned as 'wn'.")
except LookupError as e:
    # st.error is used in the UI section for NLTK errors
    print(f"NLTK LookupError during startup: {e}.")
except Exception as e:
    print(f"Unexpected error during NLTK setup: {e}")


# --- 0. Global Configuration & Components ---
DB_PATH = "alar_corpus.db"
# For Sentence Embeddings
SENTENCE_MODEL_NAME = 'all-MiniLM-L6-v2'
SEMANTIC_SIMILARITY_THRESHOLD = 0.35 # Adjusted for sentence embeddings (typically higher than TF-IDF)

KANNADA_FREQUENCY_DB_PATH = "./kannada_frequencies.db"
COMMONNESS_WEIGHT = 0.2 # Adjusted, give more weight to relevance initially
RELEVANCE_WEIGHT = 0.8

MAX_PREFILTER_CANDIDATES = 1000 # Max defs to semantically score after keyword filter
MAX_FINAL_RESULTS = 101

stemmer_global = SnowballStemmer("english")
lemmatizer_global = WordNetLemmatizer()

# --- Model Loading ---
@st.cache_resource
def load_sentence_embedding_model(model_name):
    if not sentence_transformers_available:
        print("`sentence-transformers` library not installed. Semantic search disabled.")
        return None
    try:
        print(f"Loading sentence embedding model ({model_name})...")
        model = SentenceTransformer(model_name)
        print(f"Sentence embedding model ({model_name}) loaded.")
        return model
    except Exception as e:
        print(f"Error loading sentence model '{model_name}': {e}")
        return None

sentence_model_global = load_sentence_embedding_model(SENTENCE_MODEL_NAME)


# --- Database Connection Functions ---
# (get_db_connection, check_database_tables_exist, initialize_database_connection,
#  get_frequency_db_connection, get_kannada_word_frequency functions remain the same as your script)
@st.cache_resource
def get_db_connection(db_path):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        conn.row_factory = sqlite3.Row; return conn
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
        if not definitions_table_exists: st.error("DB error: 'definitions' table missing."); return False
        if 'stemmed_words_text' not in columns: st.error("DB error: 'stemmed_words_text' missing."); return False
        if 'definition_entry' not in columns: st.error("DB error: 'definition_entry' missing."); return False
        return True
    except sqlite3.Error as e: st.error(f"SQLite error checking DB tables: {e}"); return False

def initialize_database_connection():
    if not os.path.exists(DB_PATH): 
        st.error(f"Database file '{DB_PATH}' not found! Run create_db.py."); return None
    conn = get_db_connection(DB_PATH)
    if conn is None: return None 
    if not check_database_tables_exist(conn): 
        try: conn.close()
        except: pass
        st.error("DB tables missing/schema incorrect. Re-run `create_db.py`."); return None
    return conn

@st.cache_resource
def get_frequency_db_connection(db_path):
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
        return conn
    except Exception as e: st.error(f"Error connecting to frequency DB '{db_path}': {e}"); return None

@st.cache_data 
def get_kannada_word_frequency(_freq_db_conn, kannada_word):
    if not _freq_db_conn or not kannada_word: return 0 
    try:
        cursor = _freq_db_conn.cursor()
        cursor.execute("SELECT frequency FROM word_frequencies WHERE word = ?", (kannada_word,))
        result = cursor.fetchone(); return result[0] if result else 0
    except: return 0

# --- 2. Data Access & Text Processing ---
# (get_total_entries_count, get_unique_pos_list, get_wordnet_pos_for_lemmatizer,
#  preprocess_text_for_keywords functions remain the same)
# `preprocess_text_for_tfidf` is renamed to `preprocess_text_for_lemmatization` 
# as it's primarily used for lemmatizing now.
@st.cache_data
def get_total_entries_count(_conn):
    if not _conn: return 0
    try: cursor = _conn.cursor(); cursor.execute("SELECT COUNT(id) FROM words"); return cursor.fetchone()[0]
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
    global wn, wordnet_available; 
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
    except LookupError: return [stemmer_instance.stem(cleaned_text)] 
    except: return [] 
    return sorted(list(set(stemmer_instance.stem(token) for token in tokens if token and token not in stop_words_set and len(token) > 1)))

def preprocess_text_for_lemmatization(text, lemmatizer_instance, stop_words_set): # RENAMED from preprocess_text_for_tfidf
    global wordnet_available, wn 
    if not text or not isinstance(text, str): return [] 
    if not wordnet_available or not wn: 
        # Fallback to basic tokenization and stopword removal if POS tagging/lemmatization isn't fully available
        cleaned_text = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
        try: tokens = word_tokenize(cleaned_text)
        except: return []
        return sorted(list(set(token for token in tokens if token and token not in stop_words_set and len(token) > 1)))

    cleaned_text = re.sub(r'[^a-zA-Z0-9\s-]', '', text.lower())
    try: tokens = word_tokenize(cleaned_text)
    except LookupError: return []
    except: return []
    try: tagged_tokens = nltk.pos_tag(tokens)
    except LookupError: tagged_tokens = [(token, 'NN') for token in tokens]
    except: tagged_tokens = [(token, 'NN') for token in tokens] 
    return sorted(list(set(lemmatizer_instance.lemmatize(word, pos=get_wordnet_pos_for_lemmatizer(tag)) for word, tag in tagged_tokens if word not in stop_words_set and len(word) > 1)))

# --- 3. Query Elaboration ---
# (elaborate_query_with_wordnet and elaborate_query_with_local_llm remain the same.
# They should now return: display_elaboration, keywords_for_prefilter, elaborated_text_for_embedding)
# The third output `processed_elaboration_str_for_tfidf` will now be the text to be embedded.
def elaborate_query_with_wordnet(query_text, stemmer_instance, lemmatizer_instance, stop_words_set,
                                 max_senses=1, max_synonyms_per_sense=2):
    global wordnet_available, wn
    original_query_stemmed_keywords = preprocess_text_for_keywords(query_text, stemmer_instance, stop_words_set)
    # The text for embedding will be the richer elaboration
    
    if not query_text.strip() or not wordnet_available or not wn:
        return query_text, set(original_query_stemmed_keywords), query_text # Fallback to raw query for embedding

    lookup_tokens = [word.lower() for word in word_tokenize(query_text) if word.isalpha() and word.lower() not in stop_words_set and len(word) > 1]
    if not lookup_tokens: return query_text, set(original_query_stemmed_keywords), query_text

    display_parts = [f"**Original Query:** {query_text}"]
    elaboration_texts_for_embedding_list = [query_text] # Start with original query
    keywords_for_prefilter_set = set(original_query_stemmed_keywords)

    for token in lookup_tokens:
        try:
            synsets = wn.synsets(token)
            if not synsets: continue
            defs_disp, syns_disp = [], []
            for i, synset in enumerate(synsets):
                if i < max_senses:
                    definition = synset.definition()
                    if definition: 
                        defs_disp.append(f"- {definition}")
                        elaboration_texts_for_embedding_list.append(definition) 
                    for j, lemma in enumerate(synset.lemmas()):
                        synonym = lemma.name().replace('_', ' ').lower()
                        if synonym != token and j < max_synonyms_per_sense:
                            syns_disp.append(synonym)
                            elaboration_texts_for_embedding_list.append(synonym)
                            keywords_for_prefilter_set.update(preprocess_text_for_keywords(synonym, stemmer_instance, stop_words_set))
            if defs_disp: display_parts.append(f"\n**For '{token}':**\n*Meanings:*\n" + "\n".join(defs_disp))
            if syns_disp: display_parts.append(f"*Related terms for '{token}':* {', '.join(list(set(syns_disp)))}")
        except LookupError: wordnet_available=False; break 
        except Exception: continue 
    display_elaboration_text = "\n\n".join(display_parts)
    # Join unique parts for the text to be embedded by Sentence Transformer
    final_elaboration_text_for_embedding = " ".join(list(set(elaboration_texts_for_embedding_list))) 
    return display_elaboration_text, keywords_for_prefilter_set, final_elaboration_text_for_embedding


def get_antonyms_for_query(query_text, lemmatizer_instance, stop_words_set):
    global wordnet_available, wn
    if not wordnet_available or not query_text.strip() or not wn: return set()
    # Use preprocess_text_for_lemmatization here as antonyms are about lemmas
    processed_query_lemmas = preprocess_text_for_lemmatization(query_text, lemmatizer_instance, stop_words_set)
    antonyms_set = set()
    for lemma_q in processed_query_lemmas:
        try:
            for ss in wn.synsets(lemma_q):
                for lm in ss.lemmas():
                    for ant in lm.antonyms(): antonyms_set.add(ant.name().replace('_', ' ').lower())
        except LookupError: wordnet_available=False; return set()
        except Exception: continue
    final_antonyms_lemmatized = set()
    for ant in antonyms_set: final_antonyms_lemmatized.update(preprocess_text_for_lemmatization(ant, lemmatizer_instance, stop_words_set))
    return final_antonyms_lemmatized

# --- Main Search Function with On-the-Fly Sentence Embeddings ---
def sentence_embedding_search_db(conn, freq_conn, original_query_text, selected_pos,
                                  stemmer_instance, lemmatizer_instance, stop_words_set,
                                  sentence_model_instance, # Sentence Transformer model
                                  elaboration_choice="WordNet"
                                  ):
    if not conn: st.error("Main DB connection unavailable."); return []
    if not sentence_model_instance: st.error("Sentence embedding model not loaded."); return []
    
    cleaned_original_query = original_query_text.strip()
    if not cleaned_original_query: return []

    # Step 1: Elaborate Query
    if elaboration_choice == "WordNet" and wordnet_available:
        display_elaboration, keywords_for_prefilter, text_to_embed_for_query = \
            elaborate_query_with_wordnet(cleaned_original_query, stemmer_instance, lemmatizer_instance, stop_words_set)
        with st.expander("Query elaboration set", expanded=True): st.markdown(text_to_embed_for_query)

    else: 
        display_elaboration = f"**Original Query:** {cleaned_original_query}"
        keywords_for_prefilter = set(preprocess_text_for_keywords(cleaned_original_query, stemmer_instance, stop_words_set))
        text_to_embed_for_query = cleaned_original_query # Use raw query if no elaboration

    query_antonyms_lemmatized = set()
    if wordnet_available:
        query_antonyms_lemmatized = get_antonyms_for_query(cleaned_original_query, lemmatizer_instance, stop_words_set)
    
    if display_elaboration != cleaned_original_query and display_elaboration != f"**Original Query:** {cleaned_original_query}": 
            with st.expander("Query Elaboration Context", expanded=False): st.markdown(display_elaboration)
    
    if not keywords_for_prefilter and not text_to_embed_for_query.strip():
        st.info("Could not derive effective keywords or elaboration from query."); return []

    try:
        query_embedding = sentence_model_instance.encode(text_to_embed_for_query)
    except Exception as e: st.error(f"Error encoding query text: {e}"); return []

    # Step 2: Keyword Pre-filtering (fetches raw definition_entry)
    is_pos_filter_active = selected_pos and selected_pos.lower() != "any"
    selected_pos_lower = selected_pos.lower() if is_pos_filter_active else ""
    sql_fetch_for_prefilter = "SELECT id as definition_id, word_id, definition_entry, stemmed_words_text FROM definitions WHERE stemmed_words_text IS NOT NULL"
    params_prefilter = []
    if is_pos_filter_active: sql_fetch_for_prefilter += " AND LOWER(type) = ?"; params_prefilter.append(selected_pos_lower)
    
    candidate_definitions_from_db = []
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
                                                    'raw_text': row['definition_entry']})
                if len(candidate_definitions_from_db) >= MAX_PREFILTER_CANDIDATES: break
    except sqlite3.Error as e: st.error(f"Error during keyword pre-filtering: {e}"); return []

    if not candidate_definitions_from_db: st.info("Keyword pre-filter found no candidate definitions."); return []
    
    # Step 3: On-the-fly Embedding Generation for Candidates & Similarity Calculation
    definition_texts_to_embed = [cand['raw_text'] for cand in candidate_definitions_from_db if cand['raw_text']]
    results_with_scores = [] # To store dicts with original info + relevance_score
    
    if definition_texts_to_embed:
        try:
            with st.spinner(f"Encoding {len(definition_texts_to_embed)} candidate definitions..."):
                definition_embeddings = sentence_model_instance.encode(definition_texts_to_embed)
            
            if query_embedding.ndim == 1: query_embedding_reshaped = query_embedding.reshape(1, -1)
            else: query_embedding_reshaped = query_embedding

            similarities = cosine_similarity(query_embedding_reshaped, definition_embeddings).flatten()

            idx = 0 
            for i, cand_def_info in enumerate(candidate_definitions_from_db):
                if not cand_def_info['raw_text']: continue
                relevance_score = float(similarities[idx])
                idx += 1
                if relevance_score >= SEMANTIC_SIMILARITY_THRESHOLD: # Semantic threshold
                    results_with_scores.append({**cand_def_info, 'relevance_score': relevance_score}) 
        except Exception as e: st.error(f"Error during sentence embedding/similarity: {e}"); return []
    
    if not results_with_scores: st.info(f"No definitions passed semantic similarity threshold ({SEMANTIC_SIMILARITY_THRESHOLD})."); return []

    # --- Integrate Commonness Score & Antonym Penalization ---
    # (This logic is copied and adapted from your previous TF-IDF based script's commonness integration)
    kannada_headwords_map = {}
    if freq_conn:
        candidate_word_ids_for_freq = list(set(r['word_id'] for r in results_with_scores))
        if candidate_word_ids_for_freq:
            placeholders_freq = ','.join(['?'] * len(candidate_word_ids_for_freq))
            sql_words = f"SELECT id, entry_text FROM words WHERE id IN ({placeholders_freq})"
            try:
                main_db_cursor = conn.cursor()
                main_db_cursor.execute(sql_words, candidate_word_ids_for_freq)
                for row_word in main_db_cursor.fetchall(): kannada_headwords_map[row_word['id']] = row_word['entry_text']
            except sqlite3.Error as e: st.warning(f"Could not fetch Kannada headwords for frequency: {e}")

        raw_frequencies_in_batch = []
        for res_item in results_with_scores:
            kannada_word = kannada_headwords_map.get(res_item['word_id'])
            raw_freq = get_kannada_word_frequency(freq_conn, kannada_word) if kannada_word else 0
            res_item['raw_frequency'] = raw_freq
            if raw_freq > 0: raw_frequencies_in_batch.append(raw_freq)
        
        norm_commonness_scores_map = {} 
        if raw_frequencies_in_batch: 
            log_frequencies_for_scaling = np.log1p(np.array(raw_frequencies_in_batch, dtype=float))
            if len(log_frequencies_for_scaling) > 0:
                min_log_freq = np.min(log_frequencies_for_scaling); max_log_freq = np.max(log_frequencies_for_scaling)
                for res_item in results_with_scores:
                    if res_item['raw_frequency'] > 0:
                        log_f = np.log1p(res_item['raw_frequency'])
                        if (max_log_freq - min_log_freq) > 1e-9: norm_commonness_scores_map[res_item['word_id']] = (log_f - min_log_freq) / (max_log_freq - min_log_freq)
                        elif len(log_frequencies_for_scaling) > 0 : norm_commonness_scores_map[res_item['word_id']] = 1.0 
                        else: norm_commonness_scores_map[res_item['word_id']] = 0.0
                    else: norm_commonness_scores_map[res_item['word_id']] = 0.0
            else: 
                 for res_item in results_with_scores: norm_commonness_scores_map[res_item['word_id']] = 0.0
        else: 
            for res_item in results_with_scores: norm_commonness_scores_map[res_item['word_id']] = 0.0
    
    if query_antonyms_lemmatized and wordnet_available: # Show caption if antonyms were derived
        st.caption(f"Note: Results may be penalized for antonyms like '{', '.join(list(query_antonyms_lemmatized)[:5])}...'.")

    for res_item in results_with_scores:
        penalized_relevance = res_item['relevance_score']
        if query_antonyms_lemmatized and wordnet_available:
            # Use preprocess_text_for_lemmatization for antonym check
            definition_lemmas_set = set(preprocess_text_for_lemmatization(res_item['raw_text'], lemmatizer_global, english_stopwords_global))
            if any(antonym in definition_lemmas_set for antonym in query_antonyms_lemmatized):
                penalized_relevance *= 0.1 
        res_item['penalized_relevance_score'] = penalized_relevance

        current_normalized_commonness = 0.0
        if freq_conn and res_item.get('raw_frequency', 0) > 0 :
             current_normalized_commonness = norm_commonness_scores_map.get(res_item['word_id'], 0.0)
        res_item['commonness_score_normalized_val'] = current_normalized_commonness
        
        if freq_conn and res_item.get('raw_frequency', 0) > 0:
            res_item['final_score'] = (RELEVANCE_WEIGHT * penalized_relevance) + (COMMONNESS_WEIGHT * current_normalized_commonness)
        else: 
            res_item['final_score'] = penalized_relevance
            
    results_with_scores.sort(key=lambda x: x.get('final_score', 0.0), reverse=True)
    if not results_with_scores: st.info(f"No definitions remained after all ranking stages."); return []

    # Step 4: Fetch Full Data & Format Results 
    word_top_scores_details = {} 
    distinct_word_ids_to_fetch = set()
    for r_def in results_with_scores: 
        w_id = r_def['word_id']; distinct_word_ids_to_fetch.add(w_id)
        current_final_score = r_def.get('final_score', 0.0)
        if w_id not in word_top_scores_details or current_final_score > word_top_scores_details[w_id]['final_score']:
            word_top_scores_details[w_id] = {
                'final_score': current_final_score,
                'relevance': r_def.get('penalized_relevance_score', r_def.get('relevance_score', 0.0)),
                'commonness': r_def.get('commonness_score_normalized_val', 0.0),
                'raw_frequency_debug': r_def.get('raw_frequency', 0) 
            }
        if len(distinct_word_ids_to_fetch) >= MAX_FINAL_RESULTS + 20: break 
    if not distinct_word_ids_to_fetch: return []
    final_word_ids_list_sorted = sorted(list(distinct_word_ids_to_fetch), key=lambda wid: word_top_scores_details.get(wid, {}).get('final_score', 0.0), reverse=True)
    
    placeholders = ','.join(['?'] * len(final_word_ids_list_sorted))
    sql_fetch_words = f"SELECT id, head, entry_text, phone, origin, info FROM words WHERE id IN ({placeholders})"
    sql_fetch_all_definitions = f"SELECT id, word_id, definition_entry, type FROM definitions WHERE word_id IN ({placeholders})"
    words_data_map = {}; definitions_by_word_id_map = {}
    try: 
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
            'id': word_info_dict['id'], 'entry': word_info_dict['entry_text'], 'head': word_info_dict['head'], 
            'phone': word_info_dict['phone'], 'origin': word_info_dict['origin'], 'info': word_info_dict['info'],
            'defs': definitions_by_word_id_map.get(word_id_val, []), 
            'final_score': scores.get('final_score', 0.0),
            'relevance_score_val': scores.get('relevance', 0.0),
            'commonness_score_val': scores.get('commonness', 0.0)
        })
        if len(final_results_list) >= MAX_FINAL_RESULTS: break
    return final_results_list


# --- 4. Streamlit User Interface ---
# (This section uses your provided st.title, st.markdown, and sidebar text)
# (It also includes the logic for the elaboration_choice radio button and calls the search function)
# st.set_page_config(page_title="Alar.ink Definition Search", layout="wide")

st.title("📖 [alar.ink](https://alar.ink/) Corpus English Definition Search")
st.markdown("""
Search within [alar.ink](https://alar.ink/)'s English definitions for a matching word in Kannada and filter by part of speech (type).
Consider it a makeshift English-Kannada lookup as V. Krishna and Kailash Nadh ಅವರು work on the real thing. 
Many thanks to them both for their hard work to make alar.ink possible - V. Krishna ಅವರೇ has worked on this for 50+ years!

ವಿ. ಕೃಷ್ಣ ಮತ್ತು ಕೈಲಾಶ್ ನಾದ್ ಅವರೇ:
ನಿಮ್ಮ ಪರಿಶ್ರಮಕ್ಕೆ ಹೃತ್ಪೂರ್ವಕ ಧನ್ಯವಾದಗಳು. ಈ ಅದ್ಭುತ ನಿಘಂಟು ನನಗೆ ಕನ್ನಡವನ್ನು ಕಲಿಯಲು ಅಪಾರ ನೆರವು ನೀಡಿದೆ. 
ವಿ. ಕೃಷ್ಣ ಅವರೇ, ನಿಮ್ಮ ಐವತ್ತು ವರ್ಷದ ಪ್ರಯತ್ನ ಬಗ್ಗೆ ಓದುವಾಗ ನಿಮ್ಮ ನಿದರ್ಶನವನ್ನು ಅನುಸರಿಸಲು ನನಗೆ ಪ್ರೇರಣೆಯಾಗಿದೆ. 
""")

# --- UI Messages for Resource Status ---
if not wordnet_available: 
    st.sidebar.warning("WordNet NLTK data not fully available. WordNet-based query elaboration will be basic or disabled.")
if not sentence_transformers_available:
    st.sidebar.error("`sentence-transformers` library not installed. Semantic search using sentence embeddings is disabled.")
elif sentence_model_global is None and sentence_transformers_available:
     st.sidebar.error(f"Sentence embedding model ('{SENTENCE_MODEL_NAME}') failed to load. Semantic search disabled.")


# Initialize DB connections
db_connection = initialize_database_connection()
freq_db_connection_global = get_frequency_db_connection(KANNADA_FREQUENCY_DB_PATH) # Initialize once

if db_connection is None: 
    st.warning("Application cannot proceed: Main Database not ready."); st.stop()
if freq_db_connection_global is None:
    st.sidebar.warning(f"Frequency DB ('{KANNADA_FREQUENCY_DB_PATH}') not loaded. Commonness scores will not be used in ranking.")

total_entries_in_db = get_total_entries_count(db_connection)
unique_pos_list_from_db = get_unique_pos_list(db_connection)
if total_entries_in_db == 0: 
    st.error(f"DB '{DB_PATH}' empty. Run create_db.py."); st.stop()

col1, col2 = st.columns([3, 1]) 
with col1:
    search_definition_term_input = st.text_input("Describe the definition you're looking for:", placeholder="e.g., a feeling of joy, instrument for writing")
with col2:
    selected_part_of_speech_filter = st.selectbox("Part of Speech (optional):", unique_pos_list_from_db)

# Elaboration Method Selection UI
elaboration_method_options = ["None (Raw Query)"] 
default_elaboration_idx = 0
if wordnet_available:
    elaboration_method_options.append("WordNet") # Simplified name
    if default_elaboration_idx == 0 and elaboration_method_options[0] == "None (Raw Query)": default_elaboration_idx = len(elaboration_method_options)-1

chosen_elaboration_method = st.radio(
    "Query Elaboration Method:", options=elaboration_method_options, index=default_elaboration_idx, horizontal=True,
    disabled=(len(elaboration_method_options) <= 1 and elaboration_method_options[0] == "None (Raw Query)"))
if chosen_elaboration_method == "None (Raw Query)" and (wordnet_available):
    st.caption("Using raw query for search. Select WordNet for richer context.")
elif not wordnet_available and chosen_elaboration_method != "None (Raw Query)":
    st.caption("Chosen elaboration method unavailable. Search will use raw query.")

# --- Search Execution ---
if search_definition_term_input and search_definition_term_input.strip():
    st.markdown("---") 
    
    # Determine if LLM should be used for elaboration based on radio button choice
    
    spinner_msg = "Searching with sentence embeddings..."
    if chosen_elaboration_method == "WordNet": spinner_msg = "Elaborating with WordNet & searching..."
    
    if not sentence_model_global: # Check if sentence transformer is available for the main search
        st.error("Sentence embedding model for search is not available. Cannot proceed.")
        st.stop()

    with st.spinner(spinner_msg):
        search_results_list = sentence_embedding_search_db( # Call the new search function
            db_connection, 
            freq_db_connection_global,
            search_definition_term_input, 
            selected_part_of_speech_filter,
            stemmer_global, 
            lemmatizer_global,
            english_stopwords_global,
            sentence_model_global, # Pass the loaded sentence transformer model
            elaboration_choice=chosen_elaboration_method, 
        )
    total_found_count = len(search_results_list)

    if search_results_list:
        # UI display logic for results, including new scores
        subheader_message_parts = [f"definitions semantically similar to \"{search_definition_term_input.strip()}\""]
        if chosen_elaboration_method == "WordNet" and wordnet_available: subheader_message_parts.append("(WordNet elaborated)")
        
        if selected_part_of_speech_filter and selected_part_of_speech_filter != "Any":
            subheader_message_parts.append(f"with POS \"{selected_part_of_speech_filter.capitalize()}\"")
        criteria_message = " ".join(subheader_message_parts)
        st.subheader(f"Found {total_found_count} relevant entr{'y' if total_found_count == 1 else 'ies'} for {criteria_message}:")
        if total_found_count >= MAX_FINAL_RESULTS: st.info(f"Displaying up to {MAX_FINAL_RESULTS} entries.")

        for i, entry_data_dict in enumerate(search_results_list):
            display_word = entry_data_dict.get('entry', "Unknown Entry")
            expander_pos_display = "N/A"; current_definitions_list = entry_data_dict.get("defs", [])
            # ... (expander_pos_display logic - kept from your script) ...
            if current_definitions_list:
                temp_pos_list = [d.get("type","").strip().lower() for d in current_definitions_list if d.get("type")]
                if selected_part_of_speech_filter and selected_part_of_speech_filter.lower() != "any":
                    if selected_part_of_speech_filter.lower() in temp_pos_list: expander_pos_display = selected_part_of_speech_filter.capitalize()
                    elif temp_pos_list: expander_pos_display = temp_pos_list[0].capitalize()
                elif temp_pos_list: expander_pos_display = temp_pos_list[0].capitalize()
            
            # --- Updated Score Display ---
            final_s = entry_data_dict.get('final_score', 0.0)
            relevance_s = entry_data_dict.get('relevance_score_val', 0.0) 
            commonness_s = entry_data_dict.get('commonness_score_val', 0.0)

            score_strings = [f"Overall: {final_s:.2f}"]
            # Only show sub-scores if they meaningfully differ or if commonness was actually applied
            show_relevance = True
            show_commonness = False

            if freq_db_connection_global and 'raw_frequency' in entry_data_dict: # Check if commonness was processed for this item
                # Commonness was processed if raw_frequency key exists (set during commonness calculation)
                show_commonness = True 
                # If commonness was used, relevance will likely differ from final_score
            elif abs(final_s - relevance_s) < 0.001 : # If only relevance contributed, no need to show it separately
                 show_relevance = False


            if show_relevance:
                score_strings.append(f"Rel: {relevance_s:.2f}")
            if show_commonness:
                 score_strings.append(f"Common: {commonness_s:.2f}")
            
            similarity_score_info = f"({', '.join(score_strings)})"
            # --- End of Updated Score Display ---

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
        st.info(f"No entries found for \"{search_definition_term_input.strip()}\".")
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