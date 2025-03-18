import os
import json
import torch
from transformers import BertTokenizerFast, BertForTokenClassification
import argparse

def load_model(model_path):
    """Load the BERT NER model from the specified path"""
    # Ensure model_path is absolute
    if not os.path.isabs(model_path):
        # Get the directory where the script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, model_path)
    
    print(f"Loading model from {model_path}...")
    
    # Check if path exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path not found: {model_path}")
    
    # Load tokenizer and model
    try:
        # Use BertTokenizerFast instead of BertTokenizer for offset mapping support
        tokenizer = BertTokenizerFast.from_pretrained(model_path)
        model = BertForTokenClassification.from_pretrained(model_path)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Error loading model: {e}")
    
    # Load tag mappings
    tag_mappings_path = os.path.join(model_path, "tag_mappings.json")
    if os.path.exists(tag_mappings_path):
        with open(tag_mappings_path, "r") as f:
            tag_mappings = json.load(f)
            id_to_tag = {int(k): v for k, v in tag_mappings["id_to_tag"].items()}
    elif hasattr(model.config, "id2label"):
        id_to_tag = model.config.id2label
    else:
        # Fallback: create basic mapping
        num_labels = model.config.num_labels
        id_to_tag = {i: f"TAG_{i}" for i in range(num_labels)}
    
    print(f"Model loaded successfully with {len(id_to_tag)} entity types")
    return tokenizer, model, id_to_tag

def predict_entities(text, tokenizer, model, id_to_tag):
    """Predict entities in the given text using BERT NER model"""
    if not text or not text.strip():
        return []
    
    # Define which labels should be ignored (not real named entities)
    # Based on your CSV, LABEL_8 appears to be non-entity text
    non_entity_labels = {"O", "LABEL_8"}
    
    # Handle both regular tokenizers and fast tokenizers
    try:
        # Process whole text (not pre-tokenized)
        encoded_input = tokenizer(
            text, 
            return_tensors="pt",
            padding=True,
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True
        )
        
        # Get offset mapping to map back to original text
        offset_mapping = encoded_input.pop("offset_mapping")[0].tolist()
        special_tokens_mask = encoded_input.pop("special_tokens_mask")[0].tolist()
        
        # Predict token classes
        with torch.no_grad():
            output = model(**encoded_input)
        
        # Get predicted token classes
        predictions = torch.argmax(output.logits, dim=2)[0].tolist()
        
        # Extract entities using offsets
        entities = []
        current_entity = None
        
        for i, (pred, offset, is_special) in enumerate(zip(predictions, offset_mapping, special_tokens_mask)):
            # Skip special tokens like [CLS], [SEP]
            if is_special or offset[0] == offset[1]:  # Special token or empty token
                continue
            
            # Get the tag for this token
            tag = id_to_tag.get(int(pred), "O")
            
            # Skip non-entity tags completely
            if tag in non_entity_labels:
                # End current entity if exists
                if current_entity:
                    entity_text = text[current_entity["start"]:current_entity["end"]]
                    if entity_text.strip():
                        entities.append({
                            "text": entity_text,
                            "start": current_entity["start"],
                            "end": current_entity["end"],
                            "label": current_entity["type"]
                        })
                    current_entity = None
                continue
                
            # Process based on tag type
            if tag.startswith("B-") or (current_entity is None):
                # End previous entity if any
                if current_entity:
                    entity_text = text[current_entity["start"]:current_entity["end"]]
                    if entity_text.strip():  # Only add non-empty entities
                        entities.append({
                            "text": entity_text,
                            "start": current_entity["start"],
                            "end": current_entity["end"],
                            "label": current_entity["type"]
                        })
                
                # Start new entity
                entity_type = tag[2:] if tag.startswith("B-") else tag
                current_entity = {
                    "type": entity_type,
                    "start": offset[0],
                    "end": offset[1]
                }
            elif tag.startswith("I-") and current_entity and current_entity["type"] == tag[2:]:
                # Continue current entity
                current_entity["end"] = offset[1]
            elif current_entity and current_entity["type"] == tag:
                # Continue current entity (non-BIO format)
                current_entity["end"] = offset[1]
            else:
                # End current entity if exists
                if current_entity:
                    entity_text = text[current_entity["start"]:current_entity["end"]]
                    if entity_text.strip():
                        entities.append({
                            "text": entity_text,
                            "start": current_entity["start"],
                            "end": current_entity["end"],
                            "label": current_entity["type"]
                        })
                
                # Start new entity
                entity_type = tag[2:] if tag.startswith("B-") else tag
                current_entity = {
                    "type": entity_type,
                    "start": offset[0],
                    "end": offset[1]
                }
        
        # Handle last entity if exists
        if current_entity:
            entity_text = text[current_entity["start"]:current_entity["end"]]
            if entity_text.strip():
                entities.append({
                    "text": entity_text,
                    "start": current_entity["start"],
                    "end": current_entity["end"],
                    "label": current_entity["type"]
                })
        
        # Post-process: clean up entity text and fix boundaries if needed
        processed_entities = []
        for entity in entities:
            entity_text = entity["text"].strip()
            if entity_text:
                # Find actual position in original text
                entity_start = text.find(entity_text, max(0, entity["start"] - 5))
                if entity_start != -1:
                    processed_entities.append({
                        "text": entity_text,
                        "start": entity_start,
                        "end": entity_start + len(entity_text),
                        "label": entity["label"]
                    })
                else:
                    # Fallback to original positions
                    processed_entities.append({
                        "text": entity_text,
                        "start": entity["start"],
                        "end": entity["start"] + len(entity_text),
                        "label": entity["label"]
                    })
        
        # Map numeric labels to human-readable labels if needed
        for entity in processed_entities:
            if entity["label"].startswith("LABEL_"):
                label_num = entity["label"].split("_")[1]
                # Map to entity type based on observed patterns in your CSV
                if label_num == "0":
                    entity["label"] = "LOCATION"
                elif label_num == "2":
                    entity["label"] = "ORGANIZATION"
                elif label_num == "3":
                    entity["label"] = "PERSON"
                elif label_num == "4":
                    entity["label"] = "LOCATION"  # Tower is typically a location
                elif label_num == "6":
                    entity["label"] = "ORGANIZATION"  # Inc is part of org
                elif label_num == "7":
                    entity["label"] = "PERSON"  # Last names
        
        return processed_entities
    
    except Exception as e:
        print(f"Error in entity prediction: {str(e)}")
        # Fallback to the original method if the direct method fails
        return fallback_predict_entities(text, tokenizer, model, id_to_tag, non_entity_labels)

def fallback_predict_entities(text, tokenizer, model, id_to_tag, non_entity_labels={"O", "LABEL_8"}):
    """Fallback method for entity prediction using token-by-token approach"""
    if not text.strip():
        return []
    
    # Pre-tokenize the text
    tokens = text.split()
    try:
        tokenized_inputs = tokenizer(
            tokens, 
            is_split_into_words=True, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            return_offsets_mapping=True,
            return_special_tokens_mask=True
        )
        
        # Get word_ids to map subwords to original words
        word_ids = tokenized_inputs.word_ids(batch_index=0)
        special_tokens_mask = tokenized_inputs.pop("special_tokens_mask")[0].tolist()
        offset_mapping = tokenized_inputs.pop("offset_mapping")[0].tolist()
        
        # Predict
        with torch.no_grad():
            outputs = model(**tokenized_inputs)
            predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
        
        # Map predictions to original tokens
        word_level_predictions = []
        current_word_idx = -1
        
        for token_idx, (word_idx, pred, special) in enumerate(zip(word_ids, predictions, special_tokens_mask)):
            if special:
                continue
            
            if word_idx != current_word_idx:
                current_word_idx = word_idx
                if word_idx is not None:
                    predicted_tag = id_to_tag.get(pred, "O")
                    # Only add if not in non_entity_labels
                    if predicted_tag not in non_entity_labels:
                        word_level_predictions.append((word_idx, predicted_tag))
        
        # Extract entities
        entities = []
        i = 0
        
        while i < len(word_level_predictions):
            word_idx, tag = word_level_predictions[i]
            
            # Skip non-entity tags
            if tag in non_entity_labels:
                i += 1
                continue
                
            if tag.startswith("B-") or (tag != "O" and not tag.startswith("I-")):
                entity_type = tag[2:] if tag.startswith("B-") else tag
                start_idx = word_idx
                end_idx = word_idx + 1
                
                # Find the end of this entity
                j = i + 1
                while j < len(word_level_predictions):
                    next_idx, next_tag = word_level_predictions[j]
                    if ((next_tag.startswith("I-") and next_tag[2:] == entity_type) or 
                        (next_tag == entity_type)):
                        end_idx = next_idx + 1
                        j += 1
                    else:
                        break
                
                # Get the entity text
                if start_idx < len(tokens) and end_idx <= len(tokens):
                    entity_tokens = tokens[start_idx:end_idx]
                    entity_text = " ".join(entity_tokens)
                    
                    # Find position in original text
                    start_pos = text.find(entity_text)
                    if start_pos != -1:
                        # Map labels to standard entity types
                        if entity_type.startswith("LABEL_"):
                            label_num = entity_type.split("_")[1]
                            if label_num == "0":
                                entity_type = "LOCATION"
                            elif label_num == "2":
                                entity_type = "ORGANIZATION"
                            elif label_num == "3":
                                entity_type = "PERSON"
                            elif label_num == "4":
                                entity_type = "LOCATION"
                            elif label_num == "6":
                                entity_type = "ORGANIZATION"
                            elif label_num == "7":
                                entity_type = "PERSON"
                        
                        entities.append({
                            "text": entity_text,
                            "start": start_pos,
                            "end": start_pos + len(entity_text),
                            "label": entity_type
                        })
                
                i = j
            else:
                i += 1
        
        return entities
    
    except Exception as e:
        print(f"Error in fallback entity prediction: {str(e)}")
        return []

def display_entities(entities, text):
    """Display the detected entities in a readable format"""
    if not entities:
        print("No entities found.")
        return
    
    print("\nDetected entities:")
    for i, entity in enumerate(entities):
        print(f"{i+1}. \"{entity['text']}\" - {entity['label']} (position: {entity['start']}-{entity['end']})")
    
    # Visual representation of entities in text
    print("\nText with highlighted entities:")
    last_end = 0
    highlighted_text = ""
    
    for entity in sorted(entities, key=lambda x: x['start']):
        # Add text before this entity
        highlighted_text += text[last_end:entity['start']]
        # Add the entity with highlighting
        highlighted_text += f"[{entity['text']}]({entity['label']})"
        last_end = entity['end']
    
    # Add any remaining text
    highlighted_text += text[last_end:]
    print(highlighted_text)

def main():
    parser = argparse.ArgumentParser(description='Run inference on BERT NER model')
    parser.add_argument('--model_path', type=str, default='data/models/bert_ner_model',
                        help='Path to the model directory')
    parser.add_argument('--text', type=str, 
                        help='Text to analyze for named entities')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode')
    
    args = parser.parse_args()
    
    # Load the model
    tokenizer, model, id_to_tag = load_model(args.model_path)
    
    if args.interactive:
        print("\nEntering interactive mode. Type 'exit' to quit.")
        while True:
            text = input("\nEnter text to analyze: ")
            if text.lower() == 'exit':
                break
            
            entities = predict_entities(text, tokenizer, model, id_to_tag)
            display_entities(entities, text)
    
    elif args.text:
        entities = predict_entities(args.text, tokenizer, model, id_to_tag)
        display_entities(entities, args.text)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()