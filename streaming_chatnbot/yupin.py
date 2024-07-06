import os
import time

import torch
import transformers
from dotenv import load_dotenv
from langchain.cache import SQLiteCache
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    CharacterTextSplitter,
    MarkdownHeaderTextSplitter,
    MarkdownTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.vectorstores import Qdrant
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextStreamer,
)

# llm_model = "MediaTek-Research/Breeze-7B-32k-Instruct-v1_0"
llm_model = "meta-llama/Meta-Llama-3-8B"
tokenizer = AutoTokenizer.from_pretrained(
    llm_model, trust_remote_code=True
)
# tokenizer = AutoTokenizer.from_pretrained(
#     llm_model, cache_dir="llm_model/", trust_remote_code=True
# )
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
streamer = TextStreamer(tokenizer=tokenizer, skip_prompt=True)

model = AutoModelForCausalLM.from_pretrained(
    llm_model,
    device_map="auto",
    low_cpu_mem_usage=True,  # try to limit RAM
    quantization_config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    ),  # load model in low precision to save memory & cache_dir="llm_model/",
    # attn_implementation="flash_attention_2",
)

# Building a LLM QNA chain
text_generation_pipeline = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    return_full_text=True,
    max_new_tokens=200,
    streamer=streamer,
)

huggingFacePipeline = HuggingFacePipeline(pipeline=text_generation_pipeline)

# # Loading and splitting the document
# loader = UnstructuredMarkdownLoader("CNS16190-zh_TW.md")
# data = loader.load()

# # Chunk text
# md_splits = MarkdownTextSplitter().split_documents(data)

# text_splitter = RecursiveCharacterTextSplitter(chunk_size=50, chunk_overlap=30)
# chunked_documents = text_splitter.split_documents(md_splits)

loader = PyPDFLoader('C:\\Users\\Administrator\Desktop\\台科\\wnec\\耀睿\\2024CNS16190\\streaming_chatnbot\\database\\en_303645v020101p.pdf')
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(documents)



sentence_transformer_model_root = "sentence_transformer_model"
sentence_transformer_model = "multi-qa-mpnet-base-dot-v1"

start_time = time.time()

doc_store = Qdrant.from_documents(
    chunked_documents,
    HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-mpnet-base-v2'
    ),
    location=":memory:",  # Local mode with in-memory storage only
)

print(f"vector database created in {time.time() - start_time:.2f} s")

qa = ConversationalRetrievalChain.from_llm(
    llm=huggingFacePipeline,
    retriever=doc_store.as_retriever(),
    return_source_documents=True,
    verbose=False,
)


def ask_question_with_context(qa, question, chat_history):
    query = ""
    result = qa({"question": question, "chat_history": chat_history})
    print("answer:", result["answer"])
    chat_history = [(query, result["answer"])]
    return chat_history


chat_history = []
while True:
    query = input("you: ")
    if query == "q":
        break
    chat_history = ask_question_with_context(qa, query, chat_history)
