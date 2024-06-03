
from threading import Thread

from dotenv import load_dotenv
import streamlit as st
from htmlTemplates import css, bot_template, user_template
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TextIteratorStreamer
from sentence_transformers import SentenceTransformer

from util import semantic_search, read_pdf_content, split_text, embed_text, create_faiss_index


# Retrieval parameters
TOP_K = 5
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
EMBED_MODEL_NAME = "intfloat/e5-base-v2"

# LLM parameters
LLM_MODEL_NAME = "sambanovasystems/SambaLingo-Turkish-Chat"
TEMPLATE = (
    "<|user|>\n"
    "Bağlam:{context}\n\nSoru:{instruction}</s>\n"
    "<|assistant|>\n"
)

@st.cache_resource
def load_tokenizer():
    return AutoTokenizer.from_pretrained(LLM_MODEL_NAME)

@st.cache_resource
def load_model():
    return AutoModelForCausalLM.from_pretrained(LLM_MODEL_NAME, 
                                                device_map='auto', 
                                                quantization_config=BNB_CONFIG,
                                                )

@st.cache_resource
def load_streamer():
    return TextIteratorStreamer(TOKENIZER, skip_prompt=True, skip_special_tokens=True)

@st.cache_resource
def load_embed_model():
    return SentenceTransformer(EMBED_MODEL_NAME)

@st.cache_resource
def load_embed_tokenizer():
    return AutoTokenizer.from_pretrained(EMBED_MODEL_NAME)

@st.cache_resource
def load_bnb_config():
    return BitsAndBytesConfig(
        load_in_8bit=True,
    )

BNB_CONFIG = load_bnb_config()

TOKENIZER = load_tokenizer()
MODEL = load_model()
STREAMER = load_streamer()
EMBED_MODEL = load_embed_model()
EMBED_TOKENIZER = load_embed_tokenizer()

def prepare_prompt(query, embed_model, split_texts, index):
    prepended_query = "query: " + query
    query_emb = embed_model.encode(prepended_query, normalize_embeddings=True)

    dist, ind = semantic_search(query_emb, index, top_k=TOP_K)

    context = "\n".join([split_texts[i] for i in ind[0]])

    prompt = TEMPLATE.format_map({'instruction': prepended_query, 'context': context})

    return context, prompt

def bot_template_generator_wrapper(generator):
    result = ""
    container = st.empty()
    for content in generator:
        result += content
        container.write(bot_template.replace("{{MSG}}", result),unsafe_allow_html=True)
    return result


def handle_question(question):

    for i,msg in enumerate(st.session_state.chat_history):
        if i%2==0:
            st.write(user_template.replace("{{MSG}}",msg,),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",msg),unsafe_allow_html=True)
    
    context, prompt = prepare_prompt(question, EMBED_MODEL, st.session_state.split_texts, st.session_state.index)
    st.session_state.chat_history.append(question)
    st.write(user_template.replace("{{MSG}}",question),unsafe_allow_html=True)
    inputs = TOKENIZER(prompt, return_tensors="pt")
    inputs['input_ids'] = inputs['input_ids'].to(MODEL.device)

    thread = Thread(target=MODEL.generate, kwargs=dict(inputs, 
                                                       max_new_tokens=512, 
                                                       do_sample=False, 
                                                       top_k=50, 
                                                       top_p=None, 
                                                       temperature=None, 
                                                       pad_token_id=TOKENIZER.eos_token_id, 
                                                       streamer=STREAMER,
                                                       repetition_penalty=1.0,
                                                       ),
                                                       )
    thread.start()
    response = bot_template_generator_wrapper(STREAMER)
    st.session_state.chat_history.append(response)

def main():
    load_dotenv()
    st.write(css,unsafe_allow_html=True)

    if "index" not in st.session_state:
        st.session_state.index = None

    if "split_texts" not in st.session_state:
        st.session_state.split_texts = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    
    st.header("Chat with multiple PDFs :books:")
    question=st.text_input("Ask question from your document:")
    if question:
        handle_question(question)
    with st.sidebar:
        st.subheader("Your documents")
        docs=st.file_uploader("Upload your PDF here and click on 'Process'",accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                
                #get the pdf
                raw_text = read_pdf_content(docs)
                
                #get the text chunks
                st.session_state.split_texts = split_text(raw_text, EMBED_TOKENIZER, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                
                #get the embeddings
                embeddings = embed_text(st.session_state.split_texts, EMBED_MODEL, prepend="passage: ")
                
                #create the faiss index
                st.session_state.index = create_faiss_index(embeddings)


if __name__ == '__main__':
    main()