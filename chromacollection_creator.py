import sys
import os
import tempfile
import streamlit as st

# Add the parent directory to the module search path
sys.path.append(os.path.abspath('../../'))

# Import from local packages
from pdf_processor import DocumentProcessor
from embedding_client import EmbeddingClient

# Import Task libraries
from langchain_core.documents import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

class ChromaCollectionCreator:
    def __init__(self, processor, embed_model, persistent_dir=None):
        """
        Initializes the ChromaCollectionCreator with a DocumentProcessor instance, an EmbeddingClient, and an optional persistent directory.

        :param processor: An instance of DocumentProcessor for processing PDF documents.
        :param embed_model: An instance of EmbeddingClient used for embedding text data.
        :param persistent_dir: An optional persistent directory to store the Chroma collection.
        """
        self.processor = processor
        self.embed_model = embed_model
        self.persistent_dir = persistent_dir or os.path.join(os.getcwd(), "chroma_db")
        self.db = None

    def create_chroma_collection(self):
        """
        Creates a Chroma collection from the processed documents.

        Steps:
        1. Verify if the DocumentProcessor instance has processed any documents. If none have been processed, display an error message using Streamlit's error widget.
        
        2. Split the processed documents into text chunks using `CharacterTextSplitter` from LangChain with a specified separator, chunk size, and chunk overlap to make the documents suitable for embedding and indexing.
        
        3. Use the `Chroma.from_documents` method to create a Chroma collection with the text chunks from step 2 and the class-initialized embeddings model.

        Provides feedback on the success or failure of the operation via Streamlit's UI.
        """
        # Check if documents have been processed
        if len(self.processor.pages) == 0:
            st.error("No documents found!", icon="📁")
            return

        # Initialize the text splitter
        splitter = CharacterTextSplitter(
            separator=" ",
            chunk_size=1024,
            chunk_overlap=100
        )
        texts = []

        # Split each document into text chunks and store them
        for page in self.processor.pages:
            text_chunks = splitter.split_text(page.page_content)
            for text in text_chunks:
                doc = Document(page_content=text, metadata={"source": "local"})
                texts.append(doc)

        if texts:
            st.success(f"Successfully split pages into {len(texts)} documents!", icon="✔️")

        # Create the Chroma Collection
        try:
            self.db = Chroma.from_documents(
                texts, 
                self.embed_model.client, 
                persist_directory=self.persistent_dir
            )
            st.success("Successfully created Chroma Collection!", icon="✔️")
        except Exception as e:
            st.error(f"Failed to create Chroma Collection: {e}", icon="🔔")

    def query_chroma_collection(self, query) -> Document:
        """
        Queries the Chroma collection for documents similar to the given query.

        :param query: The query string to search within the Chroma collection.
        :returns: The first matching document with its similarity score, or an error message if no matches are found.
        """
        if self.db:
            try:
                docs = self.db.similarity_search_with_relevance_scores(query)
                if docs:
                    return docs[0]
                else:
                    st.error("No matching documents found!", icon="🔔")
            except Exception as e:
                st.error(f"Error during query: {e}", icon="🔔")
        else:
            st.error("Chroma Collection has not been created!", icon="🔔")
        return None

    def as_retriever(self):
        """
        Converts the Chroma collection into a retriever format.

        :returns: A retriever object for the Chroma collection, or an error message if the collection is not created.
        """
        if self.db:
            return self.db.as_retriever()
        else:
            st.error("Chroma Collection has not been created!", icon="🔔")
            return None

if __name__ == "__main__":  
    
    # Main entry point for running the script as a standalone application.

    # Initializes the DocumentProcessor and EmbeddingClient, then creates a ChromaCollectionCreator instance
    # and processes the documents. Provides a Streamlit form for users to load data into the Chroma collection.
    
    try:
        # Initialize DocumentProcessor and ingest documents
        processor = DocumentProcessor()
        processor.ingest_documents()

        # Configure and initialize the EmbeddingClient
        embed_config = {
            "model_name": "textembedding-gecko@003",
            "project": "quizzify-160824",
            "location": "europe-west2"
        }
        embedding_client = EmbeddingClient(**embed_config)

        # Create the ChromaCollectionCreator instance with optional persistent directory
        persistent_dir = st.text_input("Enter persistent directory (optional)", value=os.path.join(os.getcwd(), "chroma_db"))
        chroma_creator = ChromaCollectionCreator(processor, embedding_client, persistent_dir=persistent_dir)

        # Streamlit form for user input
        with st.form("Load Data to Chroma"):
            st.write("Select PDFs for ingestion, then click Submit")

            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()

    except Exception as e:
        st.error(f"Error initializing the app: {e}", icon="🔔")