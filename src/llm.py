import os
import textwrap
from flask import Flask, request
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.gpt4all import GPT4All
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI 
from pdf2image import convert_from_path
import yaml

with open("config.yaml", "r") as f:
  config = yaml.safe_load(f)
DOCUMENTS_PATH = config['document']['folder']
POPPLER_PATH = config['poppler']

######################################## ROUTES ########################################
app = Flask(__name__)

@app.route('/message', methods=['POST'])
def message():
  body = request.get_json()
  response = chain.invoke(body['message'])

  print(f"===== Q: {body['message']}")
  print(f"===== A: {response['result']}")
  return {
    'message': fmt_response(response['result'])
  }


########################################################################################
def fmt_response(response: str):
  return("\n".join(textwrap.wrap(response, width=100)))

def get_model(oper):
  key = config['model']['llm'][oper]['use']
  return config['model']['llm'][oper]['list'][key]

def loader(path): 
  loader = PyPDFLoader(path)
  documents = loader.load_and_split()
  print(f" - size[{len(documents)}]")
  if len(documents) == 0: 
    return []

  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 64)
  return text_splitter.split_documents(documents)

def custom_prompt():
  custom_prompt_template = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng đúng và đủ theo văn bản cung cấp.
  Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
  Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng việt'

  Context: {context}
  Question: {question}

  """
  prompt = PromptTemplate(template=custom_prompt_template,
                          input_variables=['context', 'question'])
  return prompt

######################################## MAIN ########################################
chain = None
if __name__ == "__main__":
  print("************************************************************")
  print("*****                                                  *****")
  print("*****                CHAT BOT                          *****")
  print("*****                                                  *****")
  print("************************************************************")
  print(f"= LOAD DOCUMENTS: {DOCUMENTS_PATH}")

  texts = []
  for b_file in os.listdir(DOCUMENTS_PATH):
    path = os.path.join(DOCUMENTS_PATH, b_file)
    print(f"===== file: {path}", end="")
    texts += loader(path)
    # images = convert_from_path(path, poppler_path = POPPLER_PATH)
  print(f"{texts}")

  # embedding
  print(f"= INIT EMBEDDING MODEL: {config['model']['embedding']['sentence-transformers']}")
  embeddings = HuggingFaceEmbeddings(model_name = config['model']['embedding']['sentence-transformers'])
  db = Chroma.from_documents(texts, embeddings, persist_directory="db")

  # llm 
  print(f"= INIT LLM MODEL - MODE [{config['mode']}]")
  if config['mode'] == "LOCAL":
    model = get_model('gpt4all')
    llm = GPT4All(model=model, max_tokens=1000, backend="gptj", verbose=False)
  else:
    model = get_model('openai')
    os.environ["OPENAI_API_KEY"] = config["openai"]["apikey"]
    llm = ChatOpenAI(temperature=0, model_name=model)
  print(f"===== model: {model}")

  # chain
  print(f"= INIT CHAIN")
  chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs={'prompt': custom_prompt()}
  )

  print(f"= START CHAT BOT")
  app.run(debug=True)