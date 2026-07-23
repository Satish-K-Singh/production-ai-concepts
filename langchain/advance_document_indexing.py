import os
import uuid
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_text_splitters import HTMLSectionSplitter
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_classic.storage import InMemoryByteStore

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

#Splitting and ingesting the content of various URLs 
uk_collections = Chroma(
    collection_name="uk_granular",
    embedding_function=OpenAIEmbeddings(openai_api_key=api_key),
)
uk_collections.reset_collection()

uk_coarse_collections = Chroma(
    collection_name="uk_coarse",
    embedding_function=OpenAIEmbeddings(openai_api_key=api_key),
)
uk_coarse_collections.reset_collection()

# Reduce this list if you want to save on processing fees
uk_destinations = [
    "Cornwall", "North_Cornwall", "South_Cornwall", 
] 

wikivoyage_root_url = "https://en.wikivoyage.org/wiki"

uk_distination_urls = [f"{wikivoyage_root_url}/{destination}" for destination in uk_destinations]

for url in uk_distination_urls:
    html_loader = AsyncHtmlLoader(url)
    docs = html_loader.load()
    
    for doc in docs:
        granular_chunks = document_chunk([doc])
        uk_collections.add_documents(granular_chunks)
        coarse_chunks = coarse_document_chunk([doc])
        uk_coarse_collections.add_documents(coarse_chunks)
   
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=3000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=500)

child_collection = Chroma(
    collection_name="my_child_collection",
    embedding_function=OpenAIEmbeddings(openai_api_key=api_key),
)

child_collection.reset_collection()
byte_store = InMemoryByteStore()

doc_key = "doc_id"

multi_vector_retriever = MultiVectorRetriever( #F
    vectorstore=child_collection,
    byte_store=byte_store
)

for url in uk_distination_urls:
    html_loader = AsyncHtmlLoader(url)
    docs = html_loader.load()
    text_docs = html2text_transformer.transform_documents(docs)
    coarse_chunks = parent_splitter.split_documents(text_docs)
    coarse_chunks_ids = [str(uuid.uuid4()) for _ in coarse_chunks]
    all_granular_chunks = []
    
    all_granular_chunks = []

for i, coarse_chunk in enumerate(coarse_chunks):
    coarse_chunk_id = coarse_chunks_ids[i]
    
    # Split the coarse chunk into smaller child chunks
    granular_chunks = child_splitter.split_documents([coarse_chunk])

    # Stamp each child chunk with the parent's ID
    for granular_chunk in granular_chunks:
        granular_chunk.metadata[doc_key] = coarse_chunk_id

    # FIX: Use .extend() so we add Document objects, not lists or tuples
    all_granular_chunks.extend(granular_chunks)

# 1. Vectorstore takes a flat List[Document]
multi_vector_retriever.vectorstore.add_documents(all_granular_chunks)

# 2. Docstore mset takes a List[Tuple[str, Document]]
multi_vector_retriever.docstore.mset(list(zip(coarse_chunks_ids, coarse_chunks)))

retrieved_docs = multi_vector_retriever.invoke(
    "Cornwall Ranger")

child_docs_only =  child_collection.similarity_search(
    "Cornwall Ranger")

print("Child Collection Search Results:", child_docs_only[0])