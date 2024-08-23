import os
import sys
import json
import streamlit as st
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Add the path for custom modules
sys.path.append(os.path.abspath('../../'))

from pdf_processor import DocumentProcessor
from embedding_client import EmbeddingClient
from chromacollection_creator import ChromaCollectionCreator

from langchain_core.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

class Choice(BaseModel):
    key: str = Field(description="The key for the choice, should be one of 'A', 'B', 'C', or 'D'.")
    value: str = Field(description="The text of the choice.")

class QuestionSchema(BaseModel):
    question: str = Field(description="The text of the question.")
    choices: List[Choice] = Field(description="A list of choices for the question, each with a key and a value.")
    answer: str = Field(description="The key of the correct answer from the choices list.")
    explanation: str = Field(description="An explanation as to why the answer is correct.")

model_config = {
    "json_schema_extra": {
        "examples": [
            {
                "question": "What is the capital of France?",
                "choices": [
                    {"key": "A", "value": "Berlin"},
                    {"key": "B", "value": "Madrid"},
                    {"key": "C", "value": "Paris"},
                    {"key": "D", "value": "Rome"}
                ],
                "answer": "C",
                "explanation": "Paris is the capital of France."
            }
        ]
    }
}

class QuizGenerator:
    def __init__(self, topic=None, num_questions=1, vectorstore=None):
        self.topic = topic if topic else "General Knowledge"
        if num_questions > 10:
            raise ValueError("Number of questions cannot exceed 10.")
        self.num_questions = num_questions
        self.vectorstore = vectorstore
        self.llm = None
        self.parser = JsonOutputParser(pydantic_object=QuestionSchema)
        self.question_bank = []  # Initialize the question bank to store questions
        self.system_template = """
            You are a subject matter expert on the topic: {topic}
            
            Follow the instructions to create a quiz question:
            1. Generate a question based on the topic provided and context as key "question"
            2. Provide 4 multiple choice answers to the question as a list of key-value pairs "choices"
            3. Provide the correct answer for the question from the list of answers as key "answer"
            4. Provide an explanation as to why the answer is correct as key "explanation"
            
            {format_instructions}
            
            Context: {context}
            """

    def init_llm(self):
        try:
            self.llm = VertexAI(
                model_name="gemini-pro",
                temperature=0.8,
                max_output_tokens=500
            )
            logging.info("LLM initialized successfully.")
        except Exception as e:
            logging.error(f"Error initializing LLM: {e}")
            raise

    def generate_question_with_vectorstore(self):
        if not self.llm:
            self.init_llm()
        if not self.vectorstore:
            raise ValueError("Vectorstore not provided.")
        
        from langchain_core.runnables import RunnablePassthrough, RunnableParallel

        try:
            retriever = self.vectorstore.as_retriever()
            prompt = PromptTemplate(
                template=self.system_template,
                input_variables=["topic", "context"],
                partial_variables={"format_instructions": self.parser.get_format_instructions()},
            )
            setup_and_retrieval = RunnableParallel(
                {"context": retriever, "topic": RunnablePassthrough()}
            )
            chain = setup_and_retrieval | prompt | self.llm | self.parser
            response = chain.invoke(self.topic)
            logging.info("Question generated successfully.")
            return response
        except Exception as e:
            logging.error(f"Error generating question: {e}")
            raise

    def generate_quiz(self) -> list:
        self.question_bank = [] # Reset the question bank

        for _ in range(self.num_questions):
            try:
                question = self.generate_question_with_vectorstore()

                if self.validate_question(question):
                    logging.info("Successfully generated unique question")
                    self.question_bank.append(question)
                else:
                    logging.warning("Duplicate or invalid question detected.")
                    for i in range(3):  # Retry limit of 3 attempts
                        try:
                            question_str = self.generate_question_with_vectorstore()
                            question = json.loads(question_str)
                            if self.validate_question(question):
                                logging.info("Successfully generated unique question")
                                self.question_bank.append(question)
                                break
                            else:
                                logging.warning(f"Duplicate or invalid question detected - Attempt {i+1}")
                        except json.JSONDecodeError:
                            logging.error("Failed to decode question JSON.")
                            continue
            except Exception as e:
                logging.error(f"Error generating quiz: {e}")
                continue

        return self.question_bank
    
    def validate_question(self, question: dict) -> bool:
        if 'question' not in question or not question['question']:
            raise ValueError("The dict object must contain a non-empty 'question' key")

        is_unique = True
        for question_iterated in self.question_bank:
            if question_iterated['question'] == question['question']:
                is_unique = False
                break
        return is_unique

# Streamlit app code
if __name__ == "__main__":
    try:
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
            question_bank = None
        
            with st.form("Load Data to Chroma"):
                st.subheader("Quiz Builder")
                st.write("Select PDFs for Ingestion, the topic for the quiz, and click Generate!")
                
                topic_input = st.text_input("Topic for Generative Quiz", placeholder="Enter the topic of the document")
                questions = st.slider("Number of Questions", min_value=1, max_value=10, value=1)
                
                submitted = st.form_submit_button("Submit")
                if submitted:
                    try:
                        chroma_creator.create_chroma_collection()
                        st.write(f"Topic: {topic_input}")
                        
                        generator = QuizGenerator(topic_input, questions, chroma_creator)
                        question_bank = generator.generate_quiz()
                        question = question_bank[0] if question_bank else None
                    except Exception as e:
                        st.error(f"Error during quiz generation: {e}")
        
        if question_bank:
            screen.empty()
            with st.container():
                st.header("Generated Quiz Questions")
                for question in question_bank:
                    st.write(question)
    
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        st.error(f"An error occurred: {e}")