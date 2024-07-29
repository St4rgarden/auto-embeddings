import os
import re
import shutil
import ssl
import json
from github import Github
from nltk.tokenize import word_tokenize
from openai import OpenAI
import pinecone
import faiss
import numpy as np
from typing import List, Dict


# Attempt to download NLTK data with SSL verification disabled
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    import nltk
    nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"Warning: Failed to download NLTK data: {e}")
    print("You may need to manually download the 'punkt' dataset for NLTK.")


# 1. Data Extraction

def clone_github_repo(repo_url, local_path):
    if os.path.exists(local_path):
        print(f"Repository directory already exists. Removing: {local_path}")
        shutil.rmtree(local_path)
    os.system(f"git clone {repo_url} {local_path}")


def extract_github_data(access_token, repo_name):
    g = Github(access_token)
    try:
        repo = g.get_repo(repo_name)
    except Exception as e:
        raise ValueError(f"Failed to access repository '{repo_name}': {e}")

    local_path = f"./repos/{repo.name}"
    clone_github_repo(repo.clone_url, local_path)

    relevant_files = []
    for root, dirs, files in os.walk(local_path):
        for file in files:
            if file.endswith(('.py', '.js', '.java', '.md', '.txt', '.sol', '.ts')):
                with open(os.path.join(root, file), 'r', encoding='utf-8', errors='ignore') as f:
                    relevant_files.append(f.read())

    return "\n".join(relevant_files)


# 2. Text Processing

def preprocess_text(text):
    text = re.sub(r'(#.*$)|(//.*$)|(/\*[\s\S]*?\*/)', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text


def count_tokens(text: str) -> int:
    return len(word_tokenize(text))


def chunk_text(text, max_tokens=1600):  # Reduced from 2000 to 1000
    words = word_tokenize(text)
    chunks = []
    current_chunk = []
    current_token_count = 0

    for word in words:
        if current_token_count + 1 <= max_tokens:
            current_chunk.append(word)
            current_token_count += 1
        else:
            chunks.append(' '.join(current_chunk))
            current_chunk = [word]
            current_token_count = 1

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def generate_embedding(text: str, api_key: str) -> List[float]:
    client = OpenAI(api_key=api_key)
    try:
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=text
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return []


def generate_embeddings_in_batches(chunks: List[str], api_key: str, batch_size: int = 10) -> List[List[float]]:
    all_embeddings = []
    client = OpenAI(api_key=api_key)
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        print(f"Generating embeddings for batch {i // batch_size + 1}/{(len(chunks) - 1) // batch_size + 1}")
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002",
                input=batch
            )
            batch_embeddings = [data.embedding for data in response.data]
            all_embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Add a placeholder embedding of the correct dimension
            all_embeddings.extend([[0] * 1536 for _ in batch])
    return all_embeddings


def clear_index(index_name):
    """Clear all vectors from the specified Pinecone index."""
    try:
        index = pinecone.Index(index_name)
        index.delete(delete_all=True)
        print(f"All vectors deleted from index '{index_name}'")
    except Exception as e:
        print(f"Error clearing index '{index_name}': {e}")


def setup_pinecone(api_key, environment):
    print(f"Initializing Pinecone with environment: {environment}")
    try:
        pinecone.init(api_key=api_key, environment=environment)
        print("Pinecone initialized successfully")
    except Exception as e:
        print(f"Error initializing Pinecone: {e}")
        raise


def get_or_create_index(index_name, dimension):
    if index_name in pinecone.list_indexes():
        print(f"Index '{index_name}' already exists")
        return index_name
    else:
        print(f"Creating new index '{index_name}'")
        pinecone.create_index(index_name, dimension=dimension)
        return index_name


def create_index(index_name, dimension):
    # Ensure index name is valid
    valid_index_name = re.sub(r'[^a-z0-9-]', '-', index_name.lower())
    if valid_index_name != index_name:
        print(f"Invalid index name. Using '{valid_index_name}' instead of '{index_name}'")

    if valid_index_name not in pinecone.list_indexes():
        pinecone.create_index(valid_index_name, dimension=dimension)
    print(f"Index '{valid_index_name}' is ready")
    return valid_index_name


def insert_vectors(index_name, vectors, metadata):
    index = pinecone.Index(index_name)
    to_upsert = []
    for i, (id, vector) in enumerate(zip(metadata['ids'], vectors)):
        if len(vector) == 1536:  # Only insert vectors with the correct dimension
            to_upsert.append((id, vector, {'text': metadata['text'][i]}))
        else:
            print(f"Skipping vector {id} due to incorrect dimension: {len(vector)}")

    if to_upsert:
        index.upsert(vectors=to_upsert)
    else:
        print("No vectors to insert due to dimension mismatch")


def query_vector_db(index_name, query_vector, top_k=5):
    index = pinecone.Index(index_name)
    results = index.query(query_vector, top_k=top_k, include_metadata=True)

    if 'matches' not in results:
        print("Warning: 'matches' not found in query results")
        return []

    processed_results = []
    for match in results['matches']:
        result = {
            'id': match.get('id', 'Unknown ID'),
            'score': match.get('score', 0),
        }
        if 'metadata' in match and 'text' in match['metadata']:
            result['metadata'] = match['metadata']
        else:
            print(f"Warning: Metadata or text not found for match: {match}")
            # Fetch the vector's metadata separately if it's missing
            vector_data = index.fetch([match['id']])
            if vector_data and 'vectors' in vector_data and match['id'] in vector_data['vectors']:
                result['metadata'] = vector_data['vectors'][match['id']].get('metadata', {})
            else:
                print(f"Failed to fetch metadata for vector: {match['id']}")

        processed_results.append(result)

    return processed_results


def extract_vectors_from_pinecone(index_name: str) -> Dict[str, Dict]:
    """Extract all vectors and metadata from Pinecone index."""
    index = pinecone.Index(index_name)
    vector_data = {}
    # Fetch in batches of 1000 (Pinecone's limit)
    batch_size = 1000
    total_vector_count = index.describe_index_stats()['total_vector_count']

    # Fetch all vector IDs first
    all_ids = [f"chunk_{i}" for i in range(total_vector_count)]

    for i in range(0, total_vector_count, batch_size):
        batch_ids = all_ids[i:i + batch_size]
        results = index.fetch(ids=batch_ids)
        for id, data in results['vectors'].items():
            vector_data[id] = {
                'vector': data['values'],
                'metadata': data.get('metadata', {})
            }

    return vector_data


def create_faiss_index(vectors: List[List[float]], output_dir: str) -> None:
    """Create a FAISS index from the given vectors and save it to disk."""
    os.makedirs(output_dir, exist_ok=True)

    # Convert vectors to numpy array
    vectors_np = np.array(vectors).astype('float32')

    # Create and train the index
    dimension = vectors_np.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vectors_np)

    # Save the index to disk
    faiss.write_index(index, os.path.join(output_dir, "faiss_index"))


def save_metadata(metadata: Dict[str, Dict], output_dir: str) -> None:
    """Save metadata to disk."""
    import json
    with open(os.path.join(output_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f)


def query_faiss_index(query_vector: List[float], index_path: str, metadata_path: str, top_k: int = 5) -> List[Dict]:
    """Query the FAISS index and return results with metadata."""
    # Load the FAISS index
    index = faiss.read_index(index_path)

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Perform the query
    query_vector_np = np.array([query_vector]).astype('float32')
    distances, indices = index.search(query_vector_np, top_k)

    # Prepare results
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            'id': str(idx),
            'score': float(distances[0][i]),
            'metadata': metadata.get(str(idx), {})
        })

    return results