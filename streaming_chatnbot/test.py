from huggingface_hub import login
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import uuid
import torch
import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TextStreamer
)
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, CSVLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import Qdrant
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA

login(token='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')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['DATABASE_FOLDER'] = 'database/'
ALLOWED_EXTENSIONS = {'pdf', 'csv'}

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['DATABASE_FOLDER']):
    os.makedirs(app.config['DATABASE_FOLDER'])

model_name = 'meta-llama/Meta-Llama-3-8B'
model_config = transformers.AutoConfig.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
streamer = TextStreamer(tokenizer, skip_prompt=True)

use_4bit = True
bnb_4bit_compute_dtype = "float16"
bnb_4bit_quant_type = "nf4"
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
)

if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)

text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    repetition_penalty=1.5,
    return_full_text=False,
    max_new_tokens=100,
    streamer=streamer,
)

llama_llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
file_id = None
retrieval_chain = None
db = None

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_file(filepath):
    global retrieval_chain, db

    if filepath.endswith('.pdf'):
        loader = PyPDFLoader(filepath)
    elif filepath.endswith('.csv'):
        loader = CSVLoader(filepath, encoding='utf-8')  # 指定字符編碼
    else:
        return

    docs = loader.load_and_split()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunked_documents = text_splitter.split_documents(docs)

    if db is None:
        db = Qdrant.from_documents(chunked_documents, HuggingFaceEmbeddings(model_name='sentence-transformers/all-mpnet-base-v2'), location=":memory:")
    else:
        db.add_documents(chunked_documents)

    retriever = db.as_retriever()
    retrieval_chain = RetrievalQA.from_llm(llm=llama_llm, retriever=retriever)

def load_all_files_from_database():
    database_folder = app.config['DATABASE_FOLDER']
    for filename in os.listdir(database_folder):
        if allowed_file(filename):
            filepath = os.path.join(database_folder, filename)
            process_file(filepath)
            print(f"Processed file: {filename}")
    return 'All files in the database folder have been loaded and processed.', 200

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file_id)
        file.save(filepath)

        process_file(filepath)
        return 'File uploaded & processed successfully. You can begin querying now', 200

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
