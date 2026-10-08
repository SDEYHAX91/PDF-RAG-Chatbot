from langchain_core.vectorstores import InMemoryVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from filereader import read_multiple_pdfs

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = InMemoryVectorStore(embeddings)

DOC_PATHS = [
    "pdfs/advt.pdf",
    "pdfs/UNIX.pdf"
]

docs = read_multiple_pdfs(DOC_PATHS)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
all_splits = text_splitter.split_documents(docs)

print(f"split documentation into {len(all_splits)} chunks")

vector_store.add_documents(documents=all_splits)
print(f"indexed {len(all_splits)} documents")