import os
import json
import time
import re
from datetime import datetime, timezone
from collections import defaultdict
import unicodedata

# Third-party libraries
import fitz
import torch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import spacy
import networkx as nx

torch.manual_seed(42)
BASE_PATH = '.' 
EMBEDDING_MODEL_ID = 'sentence-transformers/all-MiniLM-L6-v2'
SPACY_MODEL_ID = 'en_core_web_sm'

def clean_text(text):
    """Cleans text by removing control characters and normalizing unicode."""
    text = unicodedata.normalize('NFKD', text)
    return "".join(c for c in text if unicodedata.category(c)[0] != 'C')

class Node:
    """Represents a node in our knowledge graph, created from a high-quality chunk."""
    def __init__(self, node_id, text, doc_name, page_num, title=""):
        self.id = node_id
        self.doc_name = doc_name
        self.page_num = page_num
        self.title = clean_text(title)
        self.text = clean_text(text)
        self.embedding = None
        self.entities = set()
        self.score = 0.0
        self.pagerank = 0.0

class ModelFactory:
    """Loads and manages all models for the pipeline."""
    def __init__(self):
        print("[ModelFactory] Initializing models...")
        self.device = torch.device("cpu")
        self.embed_model = SentenceTransformer(EMBEDDING_MODEL_ID, device=self.device)
        self.nlp = spacy.load(SPACY_MODEL_ID)
        print("[ModelFactory] Models loaded successfully.")

class LayoutAwareChunker:
    """A more robust chunker that uses multi-factor analysis for title detection."""
    
    def analyze_document_style(self, doc):
        """Dynamically learns font sizes and line lengths from the document."""
        font_counts = defaultdict(int)
        line_lengths = []
        
        for page in doc:
            blocks = page.get_text("dict")["blocks"]
            for b in blocks:
                if "lines" in b:
                    for l in b["lines"]:
                        line_text = "".join(s["text"] for s in l["spans"]).strip()
                        if line_text:
                            line_lengths.append(len(line_text.split()))
                        if "spans" in l:
                            for s in l["spans"]:
                                if s["text"].strip():
                                    font_counts[round(s["size"])] += len(s["text"])
        
        if not font_counts: return 12, {14, 16}, 30, 350
        
        body_size = max(font_counts, key=font_counts.get)
        title_sizes = {size for size in font_counts if size > body_size * 1.15}
        
        # Dynamically set chunk sizes based on document statistics
        if line_lengths:
            line_lengths_arr = np.array(line_lengths)
            # A chunk should be larger than a typical short line
            min_chunk_words = int(np.percentile(line_lengths_arr, 40)) + 5
            # A chunk shouldn't be excessively long compared to the average
            max_chunk_words = int(np.percentile(line_lengths_arr, 95)) * 3
        else:
            min_chunk_words, max_chunk_words = 30, 350

        # Ensure values are within a reasonable range
        min_chunk_words = max(20, min_chunk_words)
        max_chunk_words = min(500, max_chunk_words)

        print(f"  - Dynamic Style: BodySize={body_size}, MinChunk={min_chunk_words}, MaxChunk={max_chunk_words}")
        return body_size, title_sizes, min_chunk_words, max_chunk_words

    def chunk(self, pdf_path):
        print(f"  [Chunker] Processing: {os.path.basename(pdf_path)}")
        chunks = []
        try:
            doc = fitz.open(pdf_path)
            body_size, title_sizes, min_chunk_words, max_chunk_words = self.analyze_document_style(doc)
            
            current_title = "Overview"
            current_text = ""

            for page_num, page in enumerate(doc, 1):
                if current_text.strip():
                    chunks.append({'text': current_text.strip(), 'doc_name': os.path.basename(pdf_path), 'page_num': page_num - 1, 'title': current_title})
                    current_text = ""

                blocks = page.get_text("dict")["blocks"]
                for b in blocks:
                    if "lines" in b:
                        for l in b["lines"]:
                            line_text = " ".join([s['text'] for s in l['spans']]).strip()
                            if not line_text: continue

                            span = l['spans'][0]
                            font_size = round(span['size'])
                            is_bold = span.get("flags", 0) & 2**4
                            
                            page_width = page.rect.width
                            line_bbox = l['bbox']
                            line_center = (line_bbox[0] + line_bbox[2]) / 2
                            is_centered = abs(line_center - page_width / 2) < page_width * 0.15

                            is_title = (
                                (font_size in title_sizes or (is_bold and font_size > body_size) or is_centered) and
                                5 <= len(line_text) <= 100 and
                                len(line_text.split()) <= 12 and
                                not line_text.endswith('.') and
                                any(c.isalpha() for c in line_text)
                            )
                            
                            if is_title:
                                if current_text.strip() and len(current_text.split()) >= min_chunk_words:
                                    chunks.append({'text': current_text.strip(), 'doc_name': os.path.basename(pdf_path), 'page_num': page_num, 'title': current_title})
                                current_title = line_text
                                current_text = ""
                            else:
                                current_text += " " + line_text
                            
                            if len(current_text.split()) > max_chunk_words:
                                chunks.append({'text': current_text.strip(), 'doc_name': os.path.basename(pdf_path), 'page_num': page_num, 'title': f"{current_title} (cont.)"})
                                current_text = ""
            
            if current_text.strip() and len(current_text.split()) >= min_chunk_words:
                chunks.append({'text': current_text.strip(), 'doc_name': os.path.basename(pdf_path), 'page_num': page_num, 'title': current_title})

            doc.close()
            print(f"  [Chunker] Extracted {len(chunks)} chunks.")
            return chunks
        except Exception as e:
            print(f"  [Chunker] Error processing {pdf_path}: {e}")
            return []

class KnowledgeGraphBuilder:
    def __init__(self, model_factory):
        self.model_factory = model_factory
        self.graph = nx.Graph()
        self.nodes = {}

    def build(self, chunks):
        print("[GraphBuilder] Building knowledge graph from high-quality chunks...")
        for i, chunk_data in enumerate(chunks):
            node = Node(i, **chunk_data)
            self.nodes[i] = node
            self.graph.add_node(i, data=node)
        
        self.post_process_graph()
        return self.graph, self.nodes

    def post_process_graph(self):
        print("[GraphBuilder] Post-processing graph...")
        node_list = list(self.nodes.values())
        if not node_list: return
        
        all_texts = [f"{node.title}. {node.text}" for node in node_list]
        embeddings = self.model_factory.embed_model.encode(all_texts, show_progress_bar=False)
        
        entity_to_nodes = defaultdict(list)
        for i, node in enumerate(node_list):
            node.embedding = embeddings[i]
            spacy_doc = self.model_factory.nlp(node.text)
            node.entities = {ent.text.lower() for ent in spacy_doc.ents if len(ent.text) > 2}
            for entity in node.entities:
                entity_to_nodes[entity].append(node.id)

        for entity, node_ids in entity_to_nodes.items():
            if len(node_ids) > 1:
                for i in range(len(node_ids)):
                    for j in range(i + 1, len(node_ids)):
                        self.graph.add_edge(node_ids[i], node_ids[j])

        if self.graph.number_of_nodes() > 0:
            pagerank_scores = nx.pagerank(self.graph)
            for node_id, score in pagerank_scores.items():
                self.nodes[node_id].pagerank = score

class GraphRanker:
    def __init__(self, graph, nodes, model_factory):
        self.graph = graph
        self.nodes = nodes
        self.model_factory = model_factory

    def rank(self, persona, job):
        print("[GraphRanker] Ranking nodes with hybrid scoring model...")
        query = f"As a {persona}, I need to {job}."
        query_embedding = self.model_factory.embed_model.encode(query)
        query_keywords = {ent.text.lower() for ent in self.model_factory.nlp(job).ents}

        all_pagerank_scores = [node.pagerank for node in self.nodes.values()]
        max_pagerank = max(all_pagerank_scores) if all_pagerank_scores else 1
        if max_pagerank == 0: max_pagerank = 1

        alpha, beta, gamma = 0.7, 0.2, 0.1
        for node in self.nodes.values():
            semantic_score = cosine_similarity(node.embedding.reshape(1, -1), query_embedding.reshape(1, -1))[0][0]
            centrality_bonus = node.pagerank / max_pagerank
            entity_match_bonus = len(query_keywords & node.entities) / (len(query_keywords) + 1)
            node.score = (alpha * semantic_score) + (beta * centrality_bonus) + (gamma * entity_match_bonus)
            
        return sorted(self.nodes.values(), key=lambda n: n.score, reverse=True)

def run_pipeline(collection_path, model_factory):
    print(f"\n{'='*20} Running Pipeline for: {os.path.basename(collection_path)} {'='*20}")
    
    input_path = os.path.join(collection_path, 'challenge1b_input.json')
    with open(input_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)

    persona, job = input_data['persona']['role'], input_data['job_to_be_done']['task']
    
    pdf_dir = os.path.join(collection_path, 'pdfs')
    if not os.path.isdir(pdf_dir): pdf_dir = collection_path
    pdf_files = [os.path.join(pdf_dir, doc['filename']) for doc in input_data['documents'] if os.path.exists(os.path.join(pdf_dir, doc['filename']))]

    chunker = LayoutAwareChunker()
    all_chunks = [chunk for pdf_path in pdf_files for chunk in chunker.chunk(pdf_path)]

    if not all_chunks:
        print("[Error] No chunks were extracted.")
        return None

    graph_builder = KnowledgeGraphBuilder(model_factory)
    graph, nodes = graph_builder.build(all_chunks)

    ranker = GraphRanker(graph, nodes, model_factory)
    ranked_nodes = ranker.rank(persona, job)

    final_chunks = []
    seen_titles = set()
    for node in ranked_nodes:
        if node.title.lower() not in seen_titles:
            final_chunks.append(node)
            seen_titles.add(node.title.lower())
        if len(final_chunks) >= 5:
            break
            
    print("[Refiner] Generating refined text for top chunks...")
    query = f"As a {persona}, I need to {job}."
    query_embedding = model_factory.embed_model.encode(query)

    output_json = {
        "metadata": {
            "input_documents": [os.path.basename(p) for p in pdf_files],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now(timezone.utc).isoformat()
        },
        "extracted_sections": [], 
        "subsection_analysis": []
    }

    for i, chunk in enumerate(final_chunks):
        rank = i + 1
        
        doc = model_factory.nlp(chunk.text)
        sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip().split()) > 5]
        
        refined_text = chunk.text
        if len(sentences) > 3:
            sentence_embeddings = model_factory.embed_model.encode(sentences, show_progress_bar=False)
            
            similarities = cosine_similarity(sentence_embeddings, query_embedding.reshape(1, -1)).flatten()
            seed_index = np.argmax(similarities)
            
            start_index = max(0, seed_index - 1)
            end_index = min(len(sentences), seed_index + 2)
            
            contextual_summary = sentences[start_index:end_index]
            refined_text = " ".join(contextual_summary)

        output_json["extracted_sections"].append({
            "document": chunk.doc_name,
            "section_title": chunk.title,
            "importance_rank": rank,
            "page_number": chunk.page_num
        })
        output_json["subsection_analysis"].append({
            "document": chunk.doc_name,
            "refined_text": refined_text,
            "page_number": chunk.page_num
        })

    return output_json

if __name__ == "__main__":
    start_time = time.time()
    all_dirs = [d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d))]
    collection_dirs = [d for d in all_dirs if d.lower().startswith('collection')]
    
    if not collection_dirs:
        print(f"Error: No directories starting with 'Collection' found in '{BASE_PATH}'.")
    else:
        print(f"Found collection directories: {collection_dirs}")
        model_factory = ModelFactory()
        
        for collection in sorted(collection_dirs):
            collection_path = os.path.join(BASE_PATH, collection)
            generated_output = run_pipeline(collection_path, model_factory)
            
            if generated_output:
                output_filename = os.path.join(collection_path, 'challenge1b_output.json')
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(generated_output, f, indent=4, ensure_ascii=False)
                print(f"\n[Success] Output saved to: {output_filename}")
    
    end_time = time.time()
    print(f"\nTotal execution time: {end_time - start_time:.2f} seconds.")
