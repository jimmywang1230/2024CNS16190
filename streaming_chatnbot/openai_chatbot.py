from huggingface_hub import login
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from dotenv import load_dotenv
import uuid
import openai
from langchain_openai import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_community.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import RetrievalQA, LLMChain
from langchain.schema import BaseRetriever
from langchain.llms import OpenAI

login(token='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')
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATABASE_FOLDER'] = 'database/'
ALLOWED_EXTENSIONS = {'pdf', 'csv'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['DATABASE_FOLDER']):
    os.makedirs(app.config['DATABASE_FOLDER'])

# 使用OpenAI GPT-4聊天模型
llm = ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4o", request_timeout=60)
retrieval_chain = None
db = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(filepath):
    global retrieval_chain, db

    _, file_extension = os.path.splitext(filepath)
    file_extension = file_extension.lower()

    print(f"Processing file: {filepath}")
    
    if file_extension == '.pdf':
        loader = PyPDFLoader(filepath)
    elif file_extension == '.csv':
        loader = CSVLoader(filepath, encoding='utf-8')
    else:
        print(f"Unsupported file type: {filepath}")
        return

    docs = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(docs)

    if db is None:
        db = Qdrant.from_documents(chunked_documents, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), location=":memory:")
    else:
        db.add_documents(chunked_documents)

    retriever: BaseRetriever = db.as_retriever()
    retrieval_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    print("Database initialized and retrieval chain set")

def load_all_files_from_database():
    database_folder = app.config['DATABASE_FOLDER']
    for filename in os.listdir(database_folder):
        if allowed_file(filename):
            filepath = os.path.join(database_folder, filename)
            process_file(filepath)
            print(f"Processed file: {filename}")
    return 'All files in the database folder have been loaded and processed.'

@app.route('/upload', methods=['POST'])
def upload_file():
    global file_id
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_id}{os.path.splitext(filename)[1]}")
        file.save(filepath)

        print(f"File uploaded: {filename}")
        print(f"Filepath: {filepath}")

        process_file(filepath)
        if retrieval_chain is not None:
            print("File uploaded & processed successfully.")
            return 'File uploaded & processed successfully. You can begin querying now', 200
        else:
            print("Failed to initialize retrieval chain.")
            return 'Failed to process file.', 500

@app.route('/query', methods=['POST'])
def query():
    global retrieval_chain
    data = request.get_json()
    if not data or 'query' not in data:
        return 'Query not provided', 400
    
    query = data['query']
    response = retrieval_chain.run(query)
    
    if "Helpful Answer: " in response:
        response = response.split("Helpful Answer: ")[-1].split("\n")[0]
    elif "Explanation: " in response:
        response = response.split("Explanation: ")[-1].split("\n")[0]

    return jsonify({"response": response}), 200

if __name__ == '__main__':
    load_all_files_from_database()
    app.run(debug=True)
