# embedding_client.py

# Import libraries
from langchain_google_vertexai import VertexAIEmbeddings

class EmbeddingClient:
    """
    Task: Initialize the EmbeddingClient class to connect to Google Cloud's Vertex AI for text embeddings.

    The EmbeddingClient class is designed to initialize an embedding client with specific configurations 
    such as the model name, project, and location. The __init__ method sets up the client with the 
    provided parameters to connect to the Google Cloud Vertex AI Embeddings service, allowing it to 
    process text queries and generate embeddings.

    Steps:
    1. Set up the __init__ method to initialize the client with the `model_name`, `project`, and `location` 
       parameters. These parameters are crucial for establishing a connection to the Google Cloud Vertex AI 
       Embeddings service.

    2. Within the __init__ method, initialize the `self.client` attribute as an instance of the 
       VertexAIEmbeddings class using the provided parameters. This attribute is then used to embed the 
       text queries.

    Parameters:
    * model_name (str): The name of the Vertex AI model used for text embeddings.
    * project (str): The Project ID on Google Cloud where the embeddings model is hosted.
    * location (str): The region (location) of the Google Cloud Project.
    """
    def __init__(self, model_name, project, location):
        self.client = VertexAIEmbeddings(
            model_name=model_name,
            project=project,
            location=location
        )

    def embed_query(self, query):
        """
        Task: Embed a text query using the initialized client.
        The embed_query method takes a text query as input and returns the corresponding embedding.
        
        Steps:
        1. Use the client's embed_text method to obtain the embedding for the provided query.
        2. Return the obtained embedding.
        
        Parameters: 
        * query (str): The text query to be embedded.
        
        Returns:
        * embedding (list): The embedding of the input query.
        
        Example:
        # Example usage
        embedding_client = EmbeddingClient(
            model_name="model_name",
            project="your-project-id",
            location="your-location"
        )
        embedding = embedding_client.embed_query("Hello, world!")
        print(embedding)
        """
        embedding = self.client.embed_query(query)
        return embedding

    def embed_documents(self, documents):
        """
        Task: Embed multiple text documents using the initialized client.
        The embed_documents method takes a list of documents as input and returns the corresponding embeddings.
        
        Steps:
        1. Use the client's embed_texts method to obtain the embeddings for the provided documents.
        2. Return the obtained embeddings.
        
        Parameters: 
        * documents (list): A list of text documents to be embedded.
        
        Returns:
        * embeddings (list): The embeddings of the input documents.
        
        Example:
        # Example usage
        embedding_client = EmbeddingClient(
            model_name="model_name",
            project="your-project-id",
            location="your-location"
        )
        documents = ["Document 1", "Document 2", "Document 3"]
        embeddings = embedding_client.embed_documents(documents)
        print(embeddings)
        """
        try:
            embeddings = self.client.embed_documents(documents)
            return embeddings
        except AttributeError:
            print("Error: The method embed_texts is not defined for the client. Please ensure the client is properly initialized.")
            return []

if __name__ == "__main__":
    # Usage
    model_name = "textembedding-gecko@003"
    project = "quizzify-160824"
    location = "europe-west2"
    
    embedding_client = EmbeddingClient(model_name, project, location)
    embeddings = embedding_client.embed_query("Hello, world!")
    print(embeddings)
    print("Successfully set up and implemented the embedding client")