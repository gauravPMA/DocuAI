from flask import Flask, render_template, request, jsonify
from PyPDF2 import PdfReader
import transformers
import torch
from transformers import StoppingCriteria, StoppingCriteriaList
from langchain.llms import HuggingFacePipeline
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = '/data/ai/gaurav/flask/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
    print(f"Created upload folder: {UPLOAD_FOLDER}")

model_id = '/data/ai/gaurav/llama-2-7b-chat-hf'
hf_auth = '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'
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

# Global variables
model = None
tokenizer = None
llm = None
vectorstore = None
chain = None

def load_model():
    global model, tokenizer, llm
    if model is None or tokenizer is None:
        bnb_config = transformers.BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model_config = transformers.AutoConfig.from_pretrained(
            model_id,
            token=hf_auth
        )
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            config=model_config,
            quantization_config=bnb_config,
            device_map='auto',
            token=hf_auth
        )
        model.eval()
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_id,
            token=hf_auth
        )
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

def process_pdf(filepath):
    global vectorstore, chain
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text = extract_text_from_pdf(filepath)
    splits = text_splitter.split_text(text)
    documents = [Document(page_content=split, metadata={}) for split in splits]

    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": "cuda" if torch.cuda.is_available() else "cpu"}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    if vectorstore is None:
        vectorstore = FAISS.from_documents(documents, embeddings)
    else:
        vectorstore.add_documents(documents)

    load_model()  # Ensure model is loaded
    chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_pdf', methods=['POST'])
def upload_pdf():
    try:
        if 'pdfUpload' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        file = request.files['pdfUpload']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        if file and file.filename.lower().endswith('.pdf'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Process the uploaded PDF
            process_pdf(filepath)
            
            return jsonify({'message': 'File uploaded and processed successfully', 'filename': filename}), 200
        else:
            return jsonify({'error': 'Invalid file type'}), 400
    except Exception as e:
        app.logger.error(f"Error uploading file: {str(e)}")
        return jsonify({'error': f'Server error: {str(e)}'}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    query = request.form.get('question')
    chat_history = []
    print("Input question:", query)
    
    if vectorstore is None:
        return jsonify({'error': 'No PDF has been uploaded yet. Please upload a PDF first.'}), 400
    
    result = chain({"question": query, "chat_history": chat_history})
    prompt = "Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer. And always give the amount in Nepali Rupees (Rs : )"
    response = result['answer'].replace(prompt, "")
    print("Generated response:", response)
    return jsonify({'answer': response})

if __name__ == '__main__':
    torch.cuda.empty_cache()
    torch.cuda.memory_summary(device=None, abbreviated=False)
    app.run(debug=True, port=5010)