import os
from dotenv import load_dotenv
from utils import (
    extract_github_data,
    preprocess_text, chunk_text, generate_embeddings_in_batches,
    setup_pinecone, get_or_create_index, clear_index, insert_vectors,
    query_vector_db, generate_prompt, count_tokens
)

# Load environment variables
load_dotenv()


def main():
    try:
        # Configuration from environment variables
        github_token = os.getenv('GITHUB_TOKEN')
        openai_api_key = os.getenv('OPENAI_API_KEY')
        pinecone_api_key = os.getenv('PINECONE_API_KEY')
        pinecone_environment = os.getenv('PINECONE_ENVIRONMENT')
        github_repo = os.getenv('GITHUB_REPO')

        # Check if all required environment variables are set
        required_vars = ['GITHUB_TOKEN', 'OPENAI_API_KEY', 'PINECONE_API_KEY', 'PINECONE_ENVIRONMENT', 'GITHUB_REPO']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

        # Extract data
        print("Extracting data from GitHub...")
        github_files = extract_github_data(github_token, github_repo)

        # Process text
        print("Processing extracted text...")
        preprocessed_text = preprocess_text(github_files)
        total_tokens = count_tokens(preprocessed_text)
        print(f"Total tokens in preprocessed text: {total_tokens}")

        chunks = chunk_text(preprocessed_text, max_tokens=1000)
        print(f"Created {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            chunk_tokens = count_tokens(chunk)
            print(f"Chunk {i + 1}: {chunk_tokens} tokens")

        # Generate embeddings
        print("Generating embeddings...")
        embeddings = generate_embeddings_in_batches(chunks, openai_api_key, batch_size=10)

        # Set up vector database
        print("Setting up Pinecone vector database...")
        setup_pinecone(pinecone_api_key, pinecone_environment)

        # Get or create index
        index_name = 'docs-index'
        valid_index_name = get_or_create_index(index_name, 1536)  # 1536 is the dimension for text-embedding-ada-002

        # Clear existing vectors from the index
        print("Clearing existing vectors from the index...")
        clear_index(valid_index_name)

        # Insert vectors
        print("Inserting vectors into the database...")
        metadata = {
            'ids': [f'chunk_{i}' for i in range(len(chunks))],
            'text': chunks
        }
        insert_vectors(valid_index_name, embeddings, metadata)

        # Example query
        print("Performing an example query...")
        query = "What are the main features of the project?"
        query_embedding = generate_embeddings_in_batches([query], openai_api_key)[0]
        results = query_vector_db(valid_index_name, query_embedding)

        # Generate prompt for Claude
        print("Generating prompt for Claude...")
        if results:
            print(f"Number of results: {len(results)}")
            for i, result in enumerate(results):
                print(f"Result {i + 1}:")
                print(f"  ID: {result.get('id', 'No ID')}")
                print(f"  Score: {result.get('score', 'No score')}")
                print(f"  Metadata: {result.get('metadata', 'No metadata')}")
                print(f"  Text: {result.get('metadata', {}).get('text', 'No text')}\n")

            context = "\n".join(
                [result.get('metadata', {}).get('text', f"No text found for chunk {result['id']}") for result in
                 results])
            print("Context used for prompt generation:")
            print(context)

            if context.strip():
                final_prompt = generate_prompt(query, context)
                print("Final prompt:")
                print(final_prompt)
            else:
                print("No text content found in the results.")
        else:
            print("No results found for the query.")

    except Exception as e:
        print(f"An error occurred: {e}")
        print("Please check your environment variables and input data.")


if __name__ == "__main__":
    main()