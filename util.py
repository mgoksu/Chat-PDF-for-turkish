
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from faiss import IndexFlatL2, write_index, read_index

def read_pdf_content(docs, joiner="\n"):
    all_content = ""
    for pdf_doc in docs:
        pdf = PdfReader(pdf_doc)
        all_content += joiner.join([page_obj.extract_text() for page_obj in pdf.pages])
    return all_content

def split_text(all_content, tokenizer, 
               chunk_size=300, chunk_overlap=50, separators=["\n\n", "\n", ". ", "? ", "! "]):

    def token_length_function(text_input):
        return len(tokenizer.encode(text_input, add_special_tokens=False))

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=token_length_function,
        separators=separators
    )

    split_texts = text_splitter.split_text(all_content)
    return split_texts

def embed_text(split_texts, embed_model, prepend=""):
    if prepend != "":
        prepended_split_texts = [prepend + text for text in split_texts]
        embeddings = embed_model.encode(prepended_split_texts, normalize_embeddings=True)
    else:
        embeddings = embed_model.encode(split_texts, normalize_embeddings=True)
    return embeddings

def save_embeddings(embeddings, file_path):
    # save embeddings
    faiss_index = IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)

    write_index(faiss_index, file_path)

def read_embeddings(file_path):
    faiss_index = read_index(file_path)
    return faiss_index

def create_faiss_index(embeddings):
    faiss_index = IndexFlatL2(embeddings.shape[1])
    faiss_index.add(embeddings)
    return faiss_index

def semantic_search(embedding, index, top_k=5):
    dist, ind = index.search(embedding.reshape(1, -1), top_k)
    return dist, ind
