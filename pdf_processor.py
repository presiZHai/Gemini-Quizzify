# pdf_processor.py

import os
import tempfile
import uuid
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader

class DocumentProcessor:
    """
    This class encapsulates the functionality for processing uploaded PDF documents using Streamlit and LangChain's PyPDFLoader.
    It provides methods to render a file uploader widget, process the uploaded PDF files, extract their pages, and display the total number of pages extracted.
    """

    def __init__(self):
        self.pages = []  # List to keep track of pages from all documents

    def ingest_documents(self):
        """
        This method renders a file uploader widget for users to upload PDF documents, processes each uploaded PDF by extracting its pages, and displays the total number of pages extracted.

        Steps:
        1. Render a file uploader widget for users to upload PDF documents.
        2. Process each uploaded file:
           a. Generate a unique filename for the temporary file.
           b. Write the uploaded file to a temporary file.
           c. Load the temporary file using PyPDFLoader.
           d. Extract all pages from the PDF.
           e. Add the extracted pages to the `pages` list.
           f. Delete the temporary file.
        3. Display the total number of pages extracted.
        """
        # Step 1: Render a file uploader widget for users to upload PDF documents.
        uploaded_files = st.file_uploader(
            "Upload one or more PDF files",
            type="pdf",
            accept_multiple_files=True
        )

        if uploaded_files is not None:
            for uploaded_file in uploaded_files:
                # Step 2a: Generate a unique filename for the temporary file.
                unique_id = uuid.uuid4().hex
                original_name, file_extension = os.path.splitext(uploaded_file.name)
                temp_filename = f"{original_name}_{unique_id}{file_extension}"
                temp_filepath = os.path.join(tempfile.gettempdir(), temp_filename)

                # Step 2b: Write the uploaded file to a temporary file.
                with open(temp_filepath, "wb") as f:
                    f.write(uploaded_file.getvalue())

                # Step 2c: Load the temporary file using PyPDFLoader.
                loader = PyPDFLoader(temp_filepath)

                # Step 2d: Extract all pages from the PDF.
                document_pages = loader.load_and_split()

                # Step 2e: Add the extracted pages to the `pages` list.
                self.pages.extend(document_pages)

                # Step 2f: Delete the temporary file.
                os.unlink(temp_filepath)

            # Step 3: Display the total number of pages extracted.
            st.write(f"Total number of pages processed: {len(self.pages)}")


if __name__ == "__main__":
    # Initialize the DocumentProcessor object
    processor = DocumentProcessor()

    # Render the file uploader widget
    processor.ingest_documents()
