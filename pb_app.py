from flask import Flask, render_template, request
from PyPDF2 import PdfReader
import pdfplumber
from torch import cuda, bfloat16
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document

app = Flask(__name__)

model_id = '/data/ai/gaurav/llama-2-7b-chat-hf'
hf_auth = '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'
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

def load_model(model_id, hf_auth, device):
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )

    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        token=hf_auth  # Updated from use_auth_token to token
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        token=hf_auth  # Updated from use_auth_token to token
    )
    model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        token=hf_auth  # Updated from use_auth_token to token
    )
    return model, tokenizer

model, tokenizer = load_model(model_id, hf_auth, device)
print(f"Model loaded on {device}")

stop_list = ['\nHuman:', '\n```\n']
stop_token_ids = [torch.LongTensor(tokenizer(x)['input_ids']).to(device) for x in stop_list]

class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        return any(torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all() for stop_ids in stop_token_ids)

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    stopping_criteria=stopping_criteria,
    temperature=0.7,
    max_new_tokens=256,
    repetition_penalty=1.0
)

llm = HuggingFacePipeline(pipeline=generate_text)

def extract_text_from_pdf(pdf_file):
    reader = PdfReader(pdf_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def load_pdf_data(pdf_files):
    all_splits = []
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    for pdf_file in pdf_files:
        # Load text data
        text = extract_text_from_pdf(pdf_file)
        splits = text_splitter.split_text(text)
        for split in splits:
            all_splits.append(Document(page_content=split, metadata={}))
    return all_splits

pdf_files = [
    "/data/ai/gaurav/flask/uploads/19 annual report english.pdf",
    "/data/ai/gaurav/flask/uploads/english report_merged.pdf",
    "/data/ai/gaurav/flask/uploads/Interim FS 2nd Qtr.-Website-final.pdf"
]

all_splits = load_pdf_data(pdf_files)

model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
vectorstore = FAISS.from_documents(all_splits, embeddings)
chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.form.get('question')
    chat_history = []
    print("Input question:", query)
    result = chain({"question": query, "chat_history": chat_history})
    prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. And always give the amount in Nepali Rupees (Rs : )"
    response = result['answer'].replace(prompt, "")
    print("Generated response:", response)
    return render_template('index.html', question=query, answer=response)

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    app.run(debug=True, host='0.0.0.0',port=5010)
