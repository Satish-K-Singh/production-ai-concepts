import os
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

#Load the OpenAI API key from the .env file
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    print("Error: OPENAI_API_KEY not found in environment variables")
    print("Please create a .env file with your OpenAI API key")

else:
    print("Initializing Chroma with OpenAI embeddings...")

#Setup the Chroma collections for storing the embeddings
collection_cornwall = Chroma(
    collection_name="my_collection",
    embedding_function=OpenAIEmbeddings(openai_api_key=api_key),
)
collection_cornwall.reset_collection()

collection_coarse = Chroma(
    collection_name="my_collection_coarse",
    embedding_function=OpenAIEmbeddings(openai_api_key=api_key),
)
collection_coarse.reset_collection()

#Load the HTML content from the destination URL
destination_url = "https://en.wikivoyage.org/wiki/Cornwall"

html_lader = AsyncHtmlLoader(destination_url)
docs = html_lader.load()

#Splitting into granular chunks with the HTMLSectionSplitter
headers_to_split_on = [("h1", "Header 1"), ("h2", "Header 2")]
html_section_splitter = HTMLSectionSplitter(headers_to_split_on=headers_to_split_on)

def document_chunk(docs):
    all_chuncks = []
    for doc in docs:
        html_string = doc.page_content
        chunks = html_section_splitter.split_text(html_string)
        all_chuncks.extend(chunks)

    return all_chuncks

granular_chunks = document_chunk(docs)

#Add the granular chunks to the Chroma collection
collection_cornwall.add_documents(granular_chunks)

#Search for the term "Cornwall" in the Chroma collection
search_results = collection_cornwall.similarity_search(query="Events or festival in Cornwall", k=3)

for doc in search_results:
    print(doc)
    print("--------------------------------------------------")

#Splitting into coarse chunks with the RecursiveCharacterTextSplitter
html2text_transformer = Html2TextTransformer()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=300)  

def coarse_document_chunk(docs):
    text_docs =  html2text_transformer.transform_documents(docs)
    coarse_chunks = text_splitter.split_documents(text_docs)
    return coarse_chunks 

coarse_chunks = coarse_document_chunk(docs)
collection_coarse.add_documents(coarse_chunks)

search_results_coarse = collection_coarse.similarity_search(query="Events or festival in Cornwall", k=3)    
for doc in search_results_coarse:
    print(doc)
    print("--------------------------------------------------")

    

