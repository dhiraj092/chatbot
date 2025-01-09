from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import openai 
from dotenv import load_dotenv
import os
import shutil

load_dotenv()

openai.api_key = os.environ["OPENAI_API_KEY"]

CHROMA_PATH = "chroma"
DATA_PATH = "data"

def load_documents():
    documents = []
    for filename in os.listdir(DATA_PATH):
        if filename.endswith('.md'):
            try:
                file_path = os.path.join(DATA_PATH, filename)
                loader = TextLoader(file_path, encoding='utf-8')
                documents.extend(loader.load())
                print(f"Loaded {filename} successfully")
            except Exception as e:
                print(f"Error loading {filename}: {str(e)}")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=500,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if chunks:
        document = chunks[0]
        print("\nExample chunk content:")
        print(document.page_content)
        print("\nMetadata:", document.metadata)

    return chunks

def save_to_chroma(chunks: list[Document]):
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    db = Chroma.from_documents(
        chunks, OpenAIEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")

def main():
    print("Loading documents...")
    documents = load_documents()
    if not documents:
        print("No documents were loaded. Please check your data directory.")
        return
        
    print("\nSplitting text...")
    chunks = split_text(documents)
    
    print("\nSaving to Chroma...")
    save_to_chroma(chunks)
    print("Done!")

if __name__ == "__main__":
    main()