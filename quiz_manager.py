# Import necessary standard Python modules for system operations and JSON handling
import os  # Provides a way to interact with the operating system, especially for file paths
import sys  # Allows access to some variables used or maintained by the interpreter and to interact with the Python runtime environment
import json  # Provides methods to parse JSON strings or convert Python objects into JSON format

# Import Streamlit, a library to create and share web apps easily with Python
import streamlit as st

# Append the parent directory path (two levels up) to the system path.
# This allows the script to import modules that reside outside its current directory.
sys.path.append(os.path.abspath('../../'))

# Import custom modules for document processing, embeddings, Chroma collection creation, and quiz generation
from pdf_processor import DocumentProcessor
from embedding_client import EmbeddingClient
from chromacollection_creator import ChromaCollectionCreator
from quiz_generator import QuizGenerator

# Define a class to manage quiz operations, such as storing questions and handling question navigation
class QuizManager:
    def __init__(self, questions: list):
        """
        Task: Initialize the QuizManager class with a list of quiz questions.

        Overview:
        This task involves setting up the `QuizManager` class by initializing it with a list of quiz question objects. 
        Each quiz question object is a dictionary that includes the question text, multiple choice options, the correct answer, and an explanation. 
        The initialization process should prepare the class for managing these quiz questions, including tracking the total number of questions.

        Instructions:
        1. Store the provided list of quiz question objects in an instance variable named `questions`.
        2. Calculate and store the total number of questions in the list in an instance variable named `total_questions`.

        Parameters:
        - questions: A list of dictionaries, where each dictionary represents a quiz question along with its choices, correct answer, and an explanation.

        Note: This initialization method is crucial for setting the foundation of the `QuizManager` class, enabling it to manage the quiz questions effectively. 
        The class will rely on this setup to perform operations such as retrieving specific questions by index and navigating through the quiz.
        """
        self.questions = questions  # Store the list of questions passed during the instantiation of the class
        self.total_questions = len(self.questions)  # Calculate and store the total number of questions

    def get_question_at_index(self, index: int):
        """
        Retrieves the quiz question object at the specified index. If the index is out of bounds,
        it restarts from the beginning index.

        :param index: The index of the question to retrieve.
        :return: The quiz question object at the specified index, with indexing wrapping around if out of bounds.
        """
        # Ensure index is always within bounds using modulo arithmetic
        valid_index = index % self.total_questions  # Use modulo to wrap around the index if it's out of bounds
        return self.questions[valid_index]  # Return the question at the valid index
    
    def next_question_index(self, direction=1):
        """
        Task: Adjust the current quiz question index based on the specified direction.

        Overview:
        Develop a method to navigate to the next or previous quiz question by adjusting the `question_index` in Streamlit's session state. 
        This method should account for wrapping, meaning if advancing past the last question or moving before the first question, it should continue from the opposite end.

        Instructions:
        1. Retrieve the current question index from Streamlit's session state.
        2. Adjust the index based on the provided `direction` (1 for next, -1 for previous), using modulo arithmetic to wrap around the total number of questions.
        3. Update the `question_index` in Streamlit's session state with the new, valid index.

        Parameters:
        - direction: An integer indicating the direction to move in the quiz questions list (1 for next, -1 for previous).

        Note: Ensure that `st.session_state["question_index"]` is initialized before calling this method. 
        This navigation method enhances the user experience by providing fluid access to quiz questions.
        """
        current_question_index = st.session_state["question_index"]  # Get the current question index from Streamlit's session state
        new_index = (current_question_index + direction) % self.total_questions  # Calculate the new index with wrapping
        st.session_state["question_index"] = new_index  # Update the session state with the new question index

# Main block of code that tests generating the quiz
if __name__ == "__main__":
    
    # Configuration dictionary for the embedding model
    embed_config = {
        "model_name": "textembedding-gecko@003",  # Name of the embedding model
        "project": "quizzify-160824",  # Project name associated with the embeddings
        "location": "europe-west2"  # Geographic location for the embedding service
    }

    # Create an empty placeholder in Streamlit for dynamic content
    screen = st.empty()
    
    with screen.container():  # Use the container inside the placeholder for layout
        st.header("Quiz Builder")  # Display a header in the app
        processor = DocumentProcessor()  # Instantiate the DocumentProcessor class
        processor.ingest_documents()  # Ingest documents (assumed to be PDF or similar) using the processor
    
        # Initialize the embedding client with the given configuration
        embed_client = EmbeddingClient(**embed_config) 
    
        # Create a Chroma collection using the document processor and embedding client
        chroma_creator = ChromaCollectionCreator(processor, embed_client)
    
        question = None  # Initialize variable to store individual questions
        question_bank = None  # Initialize variable to store the entire question bank
    
        # Streamlit form to handle user inputs for quiz generation
        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")  # Display a subheader
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")  # Provide instructions
            
            # Input field for the quiz topic
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            
            # Slider to select the number of questions to generate
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            # Submit button for the form
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()  # Create a Chroma collection based on the user input
                
                st.write(topic_input)  # Display the entered topic
                
                # Instantiate the QuizGenerator with the topic and number of questions
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                
                # Generate the quiz questions and store them in question_bank
                question_bank = generator.generate_quiz()

    # If questions have been generated, display them
    if question_bank:
        screen.empty()  # Clear the placeholder
        
        # Create a new container for displaying the quiz question
        with st.container():
            st.header("Generated Quiz Question: ")  # Display header for the quiz question
            
            # Instantiate the QuizManager with the generated question bank
            quiz_manager = QuizManager(question_bank) 
            
            # Use the get_question_at_index method to fetch the first question (0th index)
            with st.form("Multiple Choice Question"):
                
                index_question = quiz_manager.get_question_at_index(0)  # Retrieve the first question
                
                # Initialize a list to store the multiple-choice options
                choices = []
                
                # Iterate over each choice in the question and unpack the data
                for choice in index_question['choices']:
                    
                    key = choice["key"]  # Extract the key (e.g., "A", "B", etc.)
                    value = choice["value"]  # Extract the corresponding value (answer text)

                    choices.append(f"{key}) {value}")  # Append the formatted choice to the list
                
                # Display the question text in the app
                st.subheader(index_question['question'])
                
                # Display the choices as a radio button group for user selection
                answer = st.radio(
                    'Choose the correct answer',
                    choices
                )
                st.form_submit_button("Submit")  # Button to submit the answer
                
                if submitted:  # If the form is submitted
                    correct_answer_key = index_question['answer']  # Retrieve the correct answer key
                    if answer.startswith(correct_answer_key):  # Check if the selected answer is correct
                        st.success("Correct!")  # Display success message if correct
                    else:
                        st.error("Incorrect!")  # Display error message if incorrect
