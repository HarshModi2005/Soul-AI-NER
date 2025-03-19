import streamlit as st
import requests
import json
import pandas as pd
import altair as alt
import base64
from collections import Counter
import plotly.express as px
import os

# Set page configuration
st.set_page_config(
    page_title="Named Entity Recognition",
    page_icon="🔍",
    layout="wide"
)

# API Configuration
API_URL = os.getenv("API_URL", "https://your-backend-service-name.onrender.com")
API_USERNAME = os.getenv("API_USERNAME", "admin")
API_PASSWORD = os.getenv("API_PASSWORD", "password123")

def get_auth_header():
    auth_str = base64.b64encode(f"{API_USERNAME}:{API_PASSWORD}".encode()).decode()
    return {"Authorization": f"Basic {auth_str}"}

def call_ner_api(text):
    """Call the NER API and return the results"""
    try:
        response = requests.post(
            f"{API_URL}/predict", 
            headers={**get_auth_header(), "Content-Type": "application/json"},
            json={"text": text}
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {str(e)}")
        return None

def highlight_entities(text, entities):
    """Highlight entities in the original text with HTML"""
    # Sort entities by start position in reverse order to avoid index issues
    entities_sorted = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    result = text
    # Update colors to match BERT entity types
    colors = {
        "PERSON": "#8ef",
        "ORG": "#faa",
        "LOC": "#afa",
        "GPE": "#afa",  # Same as LOC for BERT
        "DATE": "#e9e",
        "TIME": "#f8d",
        "MONEY": "#ad5",
        "PERCENT": "#baa",
        "PRODUCT": "#5cd",
        "EVENT": "#faf", 
        "WORK_OF_ART": "#fab",
        "FAC": "#d8f",   # FACILITY in BERT
        "NORP": "#fed",  # Nationalities, religions, political groups
        "LANGUAGE": "#adf",
        "LAW": "#dfe",
        # Add any other entity types your BERT model uses
        "MISC": "#eee",
        "PER": "#8ef",   # Alternative for PERSON
        "B-PERSON": "#8ef",  # For BIO tagging scheme
        "I-PERSON": "#8ef",
        "B-ORG": "#faa",
        "I-ORG": "#faa"
    }
    
    for entity in entities_sorted:
        start = entity["start"]
        end = entity["end"]
        label = entity["label"]
        entity_text = entity["text"]
        
        # Get color for this entity type (default gray if not in our map)
        color = colors.get(label, "#ddd")
        
        # Create the HTML for the highlighted entity
        highlighted = f'<mark style="background-color: {color};" title="{label}">{entity_text}</mark>'
        
        # Insert the highlighted entity
        result = result[:start] + highlighted + result[end:]
    
    return result

# Title and description
st.title("Named Entity Recognition")
st.markdown("Extract entities such as people, organizations, locations, and more from your text.")

# Text input area
text_input = st.text_area(
    "Enter text for entity recognition", 
    height=200,
    placeholder="Enter text here... (e.g., 'Apple is looking at buying U.K. startup for $1 billion')",
    help="Type or paste text that you want to analyze for named entities"
)

# Process button
if st.button("Recognize Entities", type="primary"):
    if text_input:
        with st.spinner("Processing..."):
            # Call the API
            result = call_ner_api(text_input)
            
            if result:
                entities = result["entities"]
                
                # Create tabs for different views
                tab1, tab2, tab3 = st.tabs(["Highlighted Text", "Entity List", "Statistics"])
                
                with tab1:
                    # Display highlighted text
                    if entities:
                        highlighted_text = highlight_entities(text_input, entities)
                        st.markdown(f"<div style='background-color: white; padding: 10px; border-radius: 5px;'>{highlighted_text}</div>", unsafe_allow_html=True)
                    else:
                        st.info("No entities found in the text.")
                
                with tab2:
                    # Display entities as a table
                    if entities:
                        df = pd.DataFrame(entities)
                        df = df.rename(columns={"text": "Entity", "label": "Type", "start": "Start", "end": "End"})
                        df = df[["Entity", "Type", "Start", "End"]]  # Reorder columns
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.info("No entities found in the text.")
                
                with tab3:
                    # Entity statistics
                    if entities:
                        # Count entities by type
                        entity_types = [entity["label"] for entity in entities]
                        type_counts = Counter(entity_types)
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Create a bar chart
                            chart_data = pd.DataFrame({
                                "Entity Type": list(type_counts.keys()),
                                "Count": list(type_counts.values())
                            })
                            
                            fig = px.bar(
                                chart_data, 
                                x="Entity Type", 
                                y="Count", 
                                title="Entity Types Distribution",
                                color="Entity Type"
                            )
                            st.plotly_chart(fig)
                        
                        with col2:
                            # Summary statistics
                            st.metric("Total Entities", len(entities))
                            st.metric("Entity Types", len(type_counts))
                            
                            # Most common entity
                            if type_counts:
                                most_common = type_counts.most_common(1)[0]
                                st.metric("Most Common Entity", f"{most_common[0]} ({most_common[1]} occurrences)")
                    else:
                        st.info("No entities found to generate statistics.")
    else:
        st.warning("Please enter some text first.")

# Add explanation of entity types at the bottom
with st.expander("Entity Type Descriptions"):
    st.markdown("""
    | Entity Type | Description |
    | --- | --- |
    | PERSON | People, including fictional characters |
    | ORG | Companies, agencies, institutions |
    | GPE | Geopolitical entity (countries, cities, states) |
    | LOC | Non-GPE locations, mountain ranges, bodies of water |
    | DATE | Dates or periods |
    | TIME | Times smaller than a day |
    | MONEY | Monetary values, including unit |
    | PRODUCT | Objects, vehicles, foods, etc. (not services) |
    | EVENT | Named hurricanes, battles, wars, sports events, etc. |
    """)

# Footer
st.markdown("---")
st.markdown("NER Model powered by BERT and served via FastAPI")