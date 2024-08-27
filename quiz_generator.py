import os
import sys
import json
import streamlit as st
from langchain_google_vertexai import VertexAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

sys.path.append(os.path.abspath('../../'))
from document_processor import DocumentProcessor
from embedding_client import EmbeddingClient
from chromacollection_creator import ChromaCollectionCreator

class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        Generates a quiz based on a given topic and number of questions. Initializes the QuizGenerator with a required topic, the num_questions for the quiz, and an optional vectorstore for querying related information.
        
        :param topic: A string representing the required topic of the quiz.
        :param num_questions: An integer representing the number of questions to generate for the quiz, up to a maximum of 10.
        :param vectorstore: An optional vectorstore instance (e.g., ChromaDB) to be used for querying information related to the quiz topic.
        """
        if not topic:
            self.topic = "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions

        self.vectorstore = vectorstore
        self.llm = None
        self.question_bank = [] # Initiate the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation for why the answer is correct or incorrect as the key "explanation."
            
            You must respond as a JSON object with the following structure:
            {{
                "question": "<question>",
                "choices": [
                    {{"key": "A", "value": "<choice>"}},
                    {{"key": "B", "value": "<choice>"}},
                    {{"key": "C", "value": "<choice>"}},
                    {{"key": "D", "value": "<choice>"}}
                ],
                "answer": "<answer key from choices list>",
                "explanation": "<explanation as to why the answer is correct or incorrect>"
            }}
            
            Context: {context}
            """
    
    def init_llm(self):
        """
        Task: Initialize the Large Language Model (LLM) for quiz question generation.
        
        Overview:
        This method prepares the LLM for generating quiz questions by configuring essential parameters such as the model name, temperature, and maximum output tokens. The LLM will be used later to generate quiz questions based on the provided topic and context retrieved from the vectorstore.
        
        Steps:
        1. Set the LLM's model name to "gemini-pro".
        2. Configure the 'temperature' parameter to control the randomness of the output. A lower temperature results in more deterministic outputs.
        3. Specify 'max_output_tokens' to limit the length of the generated text.
        4. Initialize the LLM with the specified parameters to be ready for generating quiz questions.

        Implementation:
        - Use the VertexAI class to create an instance of the LLM with the specified configurations.
        - Assign the created LLM instance to the 'self.llm' attribute for later use in question generation.
        """
        arguments = {
            "temperature": 0.8,      # Increase for less deterministic questions
            "max_output_tokens": 500
        }

        self.llm = VertexAI(
            model_name='gemini-pro',
            **arguments
        )
        
    def generate_question_with_vectorstore(self):
        """
        Task: Generate a quiz question using the topic provided and context from the vectorstore.
        
        Overview:
        This method leverages the vectorstore to retrieve relevant context for the quiz topic, then utilizes the LLM to generate a structured quiz question in JSON format. The process involves retrieving documents, creating a prompt, and invoking the LLM to generate a question.
        
        Prerequisites:
        * Ensure the LLM has been initialized using init_llm.
        * A vectorstore must be provided and accessible via self.vectorstore.

        Steps:
        a. Verify the LLM and vectorstore are initialized and available.
        b. Retrieve relevant documents or context for the quiz topic from the vectorstore.
        c. Format the retrieved context and the quiz topic into a structured prompt using the system template.
        d. Invoke the LLM with the formatted prompt to generate a quiz question.
        e. Return the generated question in the specified JSON structure.

        Implementation:
        1. Utilize RunnableParallel and RunnablePassthrough to create a chain that integrates document retrieval and topic processing.
        2. Format the system template with the topic and retrieved context to create a comprehensive prompt for the LLM.
        3. Use the LLM to generate a quiz question based on the prompt and return the structured response.

        Note: Raising a ValueError is used to handle cases where the vectorstore is not provided.
        """
        # Initialize the LLM from the 'init_llm' method if not already initialized
        if not self.llm:
            self.init_llm()
        
        # Raise an error if the vectorstore is not initialized on the class
        if not self.vectorstore:
            raise ValueError("vectorstore not provided")

        # Enable a retriever by initializing it on the VectorStore object (class) using the as_retriever() method.
        retriever = self.vectorstore.as_retriever()
        
        # Use the .from_template method on the PromptTemplate class to create a PromptTemplate using the system template.
        promt = PromptTemplate.from_template(self.system_template)

        # Use RunnableParallel to allow the retriever to get relevant documents, and 
        # RunnablePassthrough to enable chain.invoke to send self.topic to the LLM.
        setup_and_retrieval = RunnableParallel(
            {
                "context": retriever, 
                "topic": RunnablePassthrough()
            }
        )
        
        # Create a chain with the Retriever, PromptTemplate, and LLM: chain = RETRIEVER | PROMPT | LLM 
        chain = setup_and_retrieval | promt | self.llm

        # Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response
    
    def generate_quiz(self) -> list:
        """
        Task: Generate a list of unique quiz questions based on the specified topic and number of questions.

        This method orchestrates the quiz generation process by utilizing the `generate_question_with_vectorstore` 
        method to generate each question and the `validate_question` method to ensure its uniqueness 
        before adding it to the quiz.

        Steps:
            1. Initialize an empty list to store the unique quiz questions.
            2. Loop through the desired number of questions (`num_questions`), generating each question via `generate_question_with_vectorstore`.
            3. For each generated question, validate its uniqueness using `validate_question`.
            4. If the question is unique, add it to the quiz; if not, attempt to generate a new question (consider implementing a retry limit).
            5. Return the compiled list of unique quiz questions.

        Returns:
        - A list of dictionaries, where each dictionary represents a unique quiz question generated based on the topic.

        Note: This method relies on `generate_question_with_vectorstore` for question generation and `validate_question` for ensuring question uniqueness. Ensure `question_bank` is properly initialized and managed.
        """
        self.question_bank = [] # Reset the question bank

        for _ in range(self.num_questions):
            for _ in range(0, 10):  # Try maximum 10 times when the json string could not be converted to a dictionary.
                question_str = self.generate_question_with_vectorstore() # Use class method to generate question
                
                try:
                    question = json.loads(question_str) # Convert the JSON String to a dictionary
                    
                except json.JSONDecodeError:
                    print("Failed to decode question JSON.")
                    continue  # Skip this iteration if JSON decoding fails

                # Validate the question using the validate_question method
                if question and self.validate_question(question):
                    print("Successfully generated unique question")
                    self.question_bank.append(question) # Add the valid and unique question to the bank
                    break
                else:
                    print("Duplicate or invalid question detected.")
                    continue
        
        return self.question_bank
    
    def validate_question(self, question: dict) -> bool:
        """
        Task: Validate a quiz question for uniqueness within the generated quiz.

        This method checks if the provided question (as a dictionary) is unique based on its text content compared to previously generated questions stored in `question_bank`. The goal is to ensure that no duplicate questions are added to the quiz.

        Steps:
            1. Extract the question text from the provided dictionary.
            2. Iterate over the existing questions in `question_bank` and compare their texts to the current question's text.
            3. If a duplicate is found, return False to indicate the question is not unique.
            4. If no duplicates are found, return True, indicating the question is unique and can be added to the quiz.

        Parameters:
        - question: A dictionary representing the generated quiz question, expected to contain at least a "question" key.

        Returns:
        - A boolean value: True if the question is unique, False otherwise.

        Note: This method assumes `question` is a valid dictionary and `question_bank` has been properly initialized.
        """
        is_unique = True
        question_text = question['question']
        if not question_text:
            is_unique = False
        else:
            for dictionary in self.question_bank:
                if dictionary['question'] == question_text:
                   is_unique =  False
        
        return is_unique


# Test the Generating the Quiz
if __name__ == "__main__":

    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "quizzify-160824",
        "location": "europe-west2"
    }
    
    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()
    
        embed_client = EmbeddingClient(**embed_config)  
        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        question = None
        question_bank = None
    
        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                chroma_creator.create_chroma_collection()
                
                st.write(topic_input)
                
                # Test the Quiz Generator
                generator = QuizGenerator(topic_input, questions, chroma_creator)
                question_bank = generator.generate_quiz()
                question = question_bank[0]

    if question_bank:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Question: ")
            for question in question_bank:
                st.write(question)