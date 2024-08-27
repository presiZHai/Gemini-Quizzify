import os
import sys
import json
import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

# Add the path for custom modules
sys.path.append(os.path.abspath('../../')) 
from document_processor import DocumentProcessor
from embedding_client import EmbeddingClient
from chromacollection_creator import ChromaCollectionCreator

class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        """
        Initializes the QuizGenerator with a required topic, the number of questions for the quiz,
        and an optional vectorstore for querying related information.

        :param topic: A string representing the required topic of the quiz.
        :param num_questions: An integer representing the number of questions to generate for the quiz, up to a maximum
        of 10.
        :param vectorstore: An optional vectorstore instance (e.g., ChromaDB) to be used for querying information
        related to the quiz topic.
        """
        if not topic:
            self.topic =  "General Knowledge"
        else:
            self.topic = topic

        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions
        
        self.vectorstore = vectorstore
        self.llm = None
        
        # Initialize the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"
            
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
                "explanation": "<explanation as to why the answer is correct>"
            }}
            
            Context: {context}
            """

    def init_llm(self):
        """
        Task: Initialize and configures the Large Language Model (LLM) for quiz question generation.

        Overview:
        This method configures the LLM by setting parameters like model name, temperature, and max output tokens, 
        preparing it for quiz question generation using the provided topic and context from the vectorstore.

        Steps:
        1. Set the LLM's model name to "gemini-pro."
        2. Configure the 'temperature' to control output randomness, with lower values yielding more deterministic results.
        3. Specify 'max_output_tokens' to limit the length of the generated text.
        4. Initialize the LLM with these parameters for quiz question generation.

        :return: An instance or configuration for the LLM.

        Implementation:
        - Use the VertexAI class to instantiate the LLM with the configured settings.
        - Assign the LLM instance to 'self.llm' for future use in generating questions.

        Note: Ensure you have the necessary access or API keys if required by the model.
        """
        try:
            self.llm = VertexAI(
                model_name="gemini-pro",
                temperature=0.3,
                max_output_tokens=1000
            )
        except Exception as e:
            print(f"Error initializing LLM: {e}")

    def generate_question_with_vectorstore(self):
        """
        Task: Generate a quiz question using the provided topic and context from the vectorstore.

        Overview:
        This method retrieves relevant context from the vectorstore for the quiz topic and uses the LLM to 
        generate a structured quiz question in JSON format. The process involves retrieving documents, 
        creating a prompt, and invoking the LLM.

        Prerequisites:
        - Ensure the LLM is initialized via 'init_llm'.
        - A vectorstore must be provided and accessible through 'self.vectorstore'.

        Steps:
        1. Verify the LLM and vectorstore are initialized.
        2. Retrieve relevant context for the quiz topic from the vectorstore.
        3. Format the topic and context into a structured prompt using the system template.
        4. Invoke the LLM with the prompt to generate a quiz question.
        5. Return the question in JSON format.

        Implementation:
        - Use 'RunnableParallel' and 'RunnablePassthrough' to integrate document retrieval and topic processing.
        - Format the system template with the topic and retrieved context to create a prompt for the LLM.
        - Generate the quiz question with the LLM and return the structured response.

        Note: Raise a ValueError if the vectorstore is not provided.
        """
        
        self.init_llm()
        if not self.llm:
            raise Exception("Failed to initialise llm")
        
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized.")

        retriever = self.vectorstore.as_retriever()
        if not retriever:
            raise Exception("Failed to initialise retriever")
        
        system_template = "Generate a question based on the topic: {topic} and the context: {context}"
        prompt_template = PromptTemplate.from_template(system_template)
        if not prompt_template:
            raise Exception("Failed to initialise prompt_template")
        
        # RunnableParallel allows Retriever to get relevant documents
        # RunnablePassthrough allows chain.invoke to send self.topic to LLM
        setup_and_retrieval = RunnableParallel(
            {
                "context": retriever, 
                "topic": RunnablePassthrough()
            }
        )
        if not setup_and_retrieval:
           raise Exception("Failed to initialize the setup_and_retrieval")
       
        # Create a chain with the Retriever, PromptTemplate, and LLM 
        chain = setup_and_retrieval | prompt_template | self.llm
        if not chain:
           raise Exception("Failed to initialize the chain")

        # Invoke the chain with the topic as input
        response = chain.invoke(self.topic)
        return response

# Test the object
if __name__ == "__main__":
    
    embed_config = {
        "model_name": "textembedding-gecko@003",
        "project": "quizzify-160824",
        "location": "europe-west2"
    }

    st.title("Quiz Builder")

    screen = st.empty()
    with screen.container():
        st.header("Quiz Builder")
        processor = DocumentProcessor()
        processor.ingest_documents()

        embed_client = EmbeddingClient(**embed_config)  
        chroma_creator = ChromaCollectionCreator(processor, embed_client)

        question = None

        with st.form("Load Data to Chroma"):
            st.subheader("Quiz Builder")
            st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
            
            topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
            questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
            
            submitted = st.form_submit_button("Submit")
            if submitted:
                    chroma_creator.create_chroma_collection()
                    st.write(f"Topic: {topic_input}")
                    
                    # Test Quiz generator
                    generator = QuizGenerator(topic_input, questions, chroma_creator)
                    question_bank = generator.generate_question_with_vectorstore()

    if question:
        screen.empty()
        with st.container():
            st.header("Generated Quiz Questions")
            st.write(question)