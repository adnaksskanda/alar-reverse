import streamlit as st
import yaml
import re
from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("english")

# --- 1. File Loading Setup ---
# (Rest of your script remains the same)

# --- 2. Load and Cache Data from File ---
@st.cache_data # Caches the data to avoid reloading on every interaction
def load_yaml_from_file(file_path="alar.yml"):
    """Loads and parses the YAML dictionary data from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as stream:
            word_list = yaml.safe_load(stream)
        if not isinstance(word_list, list):
            st.error(f"Error: Data in '{file_path}' is not in the expected list format. Please ensure the YAML root is a list of entries.")
            return []
        # Basic validation for the structure of items
        if word_list and not all(
            isinstance(item, dict) and 
            'entry' in item and # Main word entry
            'defs' in item and isinstance(item.get('defs'), list) and # Defs list
            all(isinstance(d, dict) and 'entry' in d and 'type' in d for d in item.get('defs', [])) # Each def has entry and type
            for item in word_list
        ):
            st.warning(f"Warning: Some items in '{file_path}' may not have the expected structure (e.g., missing 'entry', 'defs', or definition 'entry'/'type' keys). The app might not display all data correctly.")
        return word_list
    except FileNotFoundError:
        st.error(f"Error: The file '{file_path}' was not found. Please create it in the same directory as the script, or check the file name.")
        return []
    except yaml.YAMLError as exc:
        st.error(f"Error parsing YAML from file '{file_path}': {exc}")
        return []
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data from file '{file_path}': {e}")
        return []


# --- 3. Search Function (Searches definition text, filters by POS) ---
def search_corpus(search_definition_query, selected_pos, dictionary_list):
    """
    Searches entries based on text within definitions and filters by part of speech (type).
    - search_definition_query: Text to search in the 'entry' field of definitions.
    - selected_pos: Part of speech to filter by (from definition 'type'). 'Any' means no POS filter.
    """
    results = []
    if not isinstance(dictionary_list, list):
        return results

    search_def_lower = search_definition_query.lower().strip() if search_definition_query else ""

    cleaned_query_text = re.sub(r'[^\w\s-]', '', search_def_lower)
    query_words = set(word for word in cleaned_query_text.split() if word)
    if not query_words:
        return []

    stemmed_query_words = set(stemmer.stem(word) for word in query_words)
    if not stemmed_query_words:
         return []

    selected_pos_lower = selected_pos.lower() if selected_pos and selected_pos != "Any" else ""

    # If no search criteria are active, return empty list
    if not search_def_lower and not selected_pos_lower:
        return []

    for item in dictionary_list: 
        if not isinstance(item, dict) or not isinstance(item.get("defs"), list):
            continue

        # Filter by definition query (if active)
        item_matches_def_text = False
        # query_words is guaranteed to be non-empty here due to the check above
        for defn_item in item["defs"]:
            if isinstance(defn_item, dict) and "_stemmed_words_list" in defn_item and isinstance(
                defn_item["_stemmed_words_list"], list):
                # definition_text_lower = defn_item["entry"].lower()
                # # Tokenize definition text similarly to query
                # cleaned_definition_text = re.sub(r'[^\w\s-]', '', definition_text_lower)
                # definition_words = set(word for word in cleaned_definition_text.split() if word)

                stemmed_definition_words = set(defn_item["_stemmed_words_list"])
                if not stemmed_definition_words:
                    continue

                if stemmed_query_words.issubset(stemmed_definition_words): # Check if all query words are in this definition's words
                    item_matches_def_text = True
                    break # Found a matching definition in this item

        if not item_matches_def_text:
            continue # Failed definition text match (discrete words), move to next item

        # Filter by part of speech (if active)
        if selected_pos_lower: 
            match_in_pos = False
            for defn_item in item["defs"]:
                if isinstance(defn_item, dict) and "type" in defn_item and isinstance(defn_item["type"], str):
                    if selected_pos_lower == defn_item["type"].strip().lower():
                        match_in_pos = True
                        break
            if not match_in_pos: 
                continue
        
        results.append(item)
        if (len(results) > 100):
            return results

    return results


# --- 4. Streamlit User Interface ---

st.title("📖 [alar.ink](https://alar.ink/) Corpus English Definition Search")
st.markdown("""
Search within [alar.ink](https://alar.ink/)'s English definitions for a matching word in Kannada and filter by part of speech (type).
Consider this a makeshift English-Kannada lookup to serve in the meantime as V. Krishna and Kailash Nadh ಅವರು work on the real thing.
\n
""")

# Load the dictionary data from alar.yml
word_dict_lst = load_yaml_from_file("alar_stemmed.yml")

if not word_dict_lst:
    st.error("Dictionary data ('alar.yml') could not be loaded or is empty. Please ensure the file exists, is correctly formatted, and contains data.")
else:
    unique_pos_list = [
        "Any", "adjective", "adverb", "conjunction", "independent clause", "interjection",
        "noun", "pr.p.", "prefix", "preposition", "prepositional phrase", "pronoun", "sentence", "suffix",
        "verb"
    ] 

    col1, col2 = st.columns([3, 2]) 

    with col1:
        search_definition_term = st.text_input("Search within definitions:", placeholder="e.g., hint, rice, sweetness")
    with col2:
        selected_part_of_speech = st.selectbox("Filter by Part of Speech (type):", unique_pos_list)

    if search_definition_term:
        st.markdown("---") 
        search_results = search_corpus(search_definition_term, selected_part_of_speech, word_dict_lst)
        total_found_count = len(search_results)

        if search_results:
            results_to_display = search_results # Display all results

            subheader_message_parts = []
            if search_definition_term:
                subheader_message_parts.append(f"definitions matching \"{search_definition_term}\"")
            if selected_part_of_speech and selected_part_of_speech != "Any":
                subheader_message_parts.append(f"part of speech \"{selected_part_of_speech.capitalize()}\"")
            
            criteria_message = " and ".join(subheader_message_parts)
            if not criteria_message: 
                criteria_message = "your criteria"

            st.subheader(f"Found {total_found_count} entr{'y' if total_found_count == 1 else 'ies'} where {criteria_message}:")
            if total_found_count > 100:
                st.subheader("Search was too broad, max results returned is 101.")

            for i, entry_data in enumerate(results_to_display):
                display_word = entry_data.get('entry', "Unknown Entry") 

                expander_pos = "N/A" 
                if entry_data.get("defs") and isinstance(entry_data["defs"], list) and len(entry_data["defs"]) > 0:
                    if selected_part_of_speech and selected_part_of_speech != "Any":
                        has_selected_pos_in_defs = any(
                            isinstance(d, dict) and d.get("type", "").strip().lower() == selected_part_of_speech.lower()
                            for d in entry_data["defs"]
                        )
                        if has_selected_pos_in_defs:
                            expander_pos = selected_part_of_speech.capitalize()
                        else: 
                            first_def = entry_data["defs"][0]
                            if isinstance(first_def, dict) and first_def.get("type"):
                                expander_pos = first_def.get("type")
                    else: 
                        first_def = entry_data["defs"][0]
                        if isinstance(first_def, dict) and first_def.get("type"):
                            expander_pos = first_def.get("type")
                
                with st.expander(f"{display_word} ({expander_pos})", expanded=(i == 0)): # Expand first result
                    if entry_data.get('head'):
                        st.markdown(f"**Head:** {entry_data.get('head')}")
                    if entry_data.get('phone'):
                        st.markdown(f"**Phonetic:** {entry_data.get('phone')}")
                    if entry_data.get('origin'):
                        st.markdown(f"**Origin:** {entry_data.get('origin')}")
                    if entry_data.get('info'):
                        st.markdown(f"**Info:** {entry_data.get('info')}")
                    
                    st.markdown("**Definitions:**")
                    if "defs" in entry_data and entry_data["defs"]:
                        for def_idx, definition in enumerate(entry_data["defs"]):
                            if isinstance(definition, dict):
                                def_text = definition.get('entry', 'No definition entry.')
                                def_type = definition.get('type', 'N/A')
                                st.markdown(f"- **{def_text}** (*{def_type}*)")
                            else:
                                st.markdown("- Invalid definition format")
                            if def_idx < len(entry_data["defs"]) - 1: st.markdown("---")
                    else:
                        st.write("No definitions listed for this entry.")
        elif search_definition_term: 
            st.info(f"No entries found matching your criteria.")
    else:
        st.info("Enter a term to search within definitions and/or select a part of speech to filter.")

st.sidebar.header("About")
st.sidebar.info(
    "This application allows you to search within dictionary definitions "
    "and filter by part of speech (type).\n\n"
    "It loads data from an the alar.ink corpus created by V. Krishna and Kailash Nadh."
)
st.sidebar.markdown("---")
st.sidebar.markdown(f"Total entries loaded from alar.ink: **{len(word_dict_lst) if word_dict_lst else 0}**")
