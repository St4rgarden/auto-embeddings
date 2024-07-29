import os
import re
import requests
from bs4 import BeautifulSoup
import PyPDF2
import nltk
from nltk.tokenize import sent_tokenize
from github import Github
from openai import OpenAI
import pinecone

nltk.download('punkt', quiet=True)


# 1. Data Extraction

def clone_github_repo(repo_url, local_path):
    os.system(f"git clone {repo_url} {local_path}")


def extract_github_data(access_token, repo_name):
    g = Github(access_token)
    repo = g.get_repo(repo_name)

    clone_github_repo(repo.clone_url, f"./repos/{repo.name}")

    relevant_files = []
    for root, dirs, files in os.walk(f"./repos/{repo.name}"):
        for file in files:
            if file.endswith(('.py', '.js', '.java', '.md', '.txt')):
                relevant_files.append(os.path.join(root, file))

    return relevant_files


def scrape_web_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()


def extract_pdf_text(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
    return text


# 2. Text Processing

def preprocess_text(text):
    text = re.sub(r'(#.*$)|(//.*$)|(/\*[\s\S]*?\*/)', '', text, flags=re.MULTILINE)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    return text


def chunk_text(text, max_chunk_size=1000):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_chunk_size:
            current_chunk += sentence + " "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


# 3. Vector Embedding

def generate_embedding(text, api_key):
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding


# 4. Database Creation

def setup_pinecone(api_key, environment):
    pinecone.init(api_key=api_key, environment=environment)


def create_index(index_name, dimension):
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(index_name, dimension=dimension)


def insert_vectors(index_name, vectors, metadata):
    index = pinecone.Index(index_name)
    index.upsert(vectors=zip(metadata['ids'], vectors), metadata=metadata)


# 5. Integration with Claude 3.5 Sonnet

def query_vector_db(index_name, query_vector, top_k=5):
    index = pinecone.Index(index_name)
    results = index.query(query_vector, top_k=top_k, include_metadata=True)
    return results


def generate_prompt(query, context):
    prompt = f"""
    Context information:
    {context}

    Human: {query}

    Assistant: Based on the context provided, I'll do my best to answer your question.
    """
    return prompt