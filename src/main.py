import os
from dotenv import load_dotenv
from utils import (
    extract_github_data, scrape_web_page, extract_pdf_text,
    preprocess_text, chunk_text, generate_embedding,
    setup_pinecone, create_index, insert_vectors,
    query_vector_db, generate_prompt
)

# Load environment variables
load_dotenv()


def main():
    # Configuration from environment variables
    github_token = os.getenv('GITHUB_TOKEN')
    openai_api_key = os.getenv('OPENAI_API_KEY')
    pinecone_api_key = os.getenv('PINECONE_API_KEY')
    pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')

    # Check if all required environment variables are set
    required_vars = ['GITHUB_TOKEN', 'OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    # Extract data
    github_files = extract_github_data(github_token, 'owner/repo')
    web_text = scrape_web_page('https://example.com/docs')
    pdf_text = extract_pdf_text('path/to/document.pdf')

    # Process text
    all_text = "\n".join([github_files, web_text, pdf_text])
    preprocessed_text = preprocess_text(all_text)
    chunks = chunk_text(preprocessed_text)

    # Generate embeddings
    embeddings = [generate_embedding(chunk, openai_api_key) for chunk in chunks]

    # Set up vector database
    setup_pinecone(pinecone_api_key, pinecone_environment)
    create_index('docs_index', 1536)  # 1536 is the dimension for text-embedding-ada-002

    # Insert vectors
    metadata = {
        'ids': [f'chunk_{i}' for i in range(len(chunks))],
        'text': chunks
    }
    insert_vectors('docs_index', embeddings, metadata)

    # Example query
    query = "What are the main features of the project?"
    query_embedding = generate_embedding(query, openai_api_key)
    results = query_vector_db('docs_index', query_embedding)

    # Generate prompt for Claude
    context = "\n".join([result['metadata']['text'] for result in results['matches']])
    final_prompt = generate_prompt(query, context)

    print(final_prompt)


if __name__ == "__main__":
    main()