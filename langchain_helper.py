import os
from dotenv import load_dotenv
from langchain.llms import GooglePalm
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

load_dotenv()

llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0)
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='car_rental_faq.csv', source_column='Prompt')
    data = loader.load()
    vectordb = FAISS.from_documents(documents=data, embedding=instructor_embeddings)
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.7)

    prompt_template = """Given the following context and question, generate an answer based on this context only. In the answer try to provide as much text as possible from "resonse" section in the source document context without making much changes. If the answer is not found in the context, kindly state "I don't know." Do not try to make up an answer."

    CONTEXT: {context}

    QUESTION: {question}"""

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    chain = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})
    return chain

chain = get_qa_chain()
print(chain("I have a debit card and an international driver's license. Can I rent a car?"))