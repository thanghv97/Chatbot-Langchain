import os
import textwrap
from config import Config
from flask import Flask, request
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms.gpt4all import GPT4All
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI 

config = Config("config.yaml")
######################################## ROUTES ########################################
app = Flask(__name__)

@app.route('/message', methods=['POST'])
def message():
  body = request.get_json()
  print(f"===== Q: {body['message']}")

  response = chain.invoke(body['message'])

  if memory:
    print(f"===== H: {memory.load_memory_variables({})}")
  print(f"===== A: {response['result']}")
  return {
    'message': response['result']
  }


########################################################################################
def fmt_response(response: str):
  return("\n".join(textwrap.wrap(response, width=100)))

def load_pdf(p_folder):
  print(f"= LOAD DOCUMENTS: {p_folder}")

  texts = []
  for b_file in os.listdir(p_folder):
    path = os.path.join(p_folder, b_file)
    print(f"===== file: {path}", end="")

    loader = PyPDFLoader(path)
    documents = loader.load_and_split()
    print(f" - size[{len(documents)}]")

    texts += documents

  return texts

def chunk_text(document):
  print(f"= CHUNK TEXT", end="")

  text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1024, chunk_overlap = 64)
  texts = text_splitter.split_documents(documents)
  print(f" - size[{len(texts)}]")

  return texts

def custom_prompt_history():
  custom_prompt_template = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng đúng và đủ theo văn bản cung cấp.
  Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
  Nếu người dùng bảo trả lời sai, hãy kiểm tra lại câu trả lời theo thông tin cung cấp
  Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng việt

  Lịch sử chat: {chat_history}

  Thông tin: {context}
  iber: {question}
  """
  prompt = PromptTemplate(template=custom_prompt_template,
                          input_variables=['context', 'question', 'chat_history'])
  return prompt

def custom_prompt():
  custom_prompt_template = """Sử dụng các thông tin sau đây để trả lời câu hỏi của người dùng đúng và đủ theo văn bản cung cấp.
  Nếu bạn không biết câu trả lời, chỉ cần nói rằng bạn không biết, đừng cố bịa ra câu trả lời.
  Nếu người dùng bảo trả lời sai, hãy kiểm tra lại câu trả lời theo thông tin cung cấp
  Tất cả câu trả lời của bạn đều phải trả lời bằng tiếng việt

  Thông tin: {context}
  iber: {question}
  """
  prompt = PromptTemplate(template=custom_prompt_template,
                          input_variables=['context', 'question'])
  return prompt

######################################## MAIN ########################################
chain = None
memory = None
if __name__ == "__main__":
  print("************************************************************")
  print("*****                                                  *****")
  print("*****                CHAT BOT                          *****")
  print("*****                                                  *****")
  print("************************************************************")

  # load pdf
  documents = load_pdf(config.document.path)

  # chunk text
  texts = chunk_text(documents)

  # embedding
  print(f"= INIT EMBEDDING MODEL: {config.model.m_embedding}")
  embeddings = HuggingFaceEmbeddings(model_name = config.model.m_embedding)
  db = Chroma.from_documents(texts, embeddings, persist_directory="db")

  # llm 
  print(f"= INIT LLM MODEL - MODE [{config.mode}]")
  if config.mode == "LOCAL":
    model = config.model.m_llm['gpt4all']
    llm = GPT4All(model=model, max_tokens=1000, backend="gptj", verbose=False)
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)
  else:
    model = config.model.m_llm['openai']
    os.environ["OPENAI_API_KEY"] = config.config["openai"]["apikey"]
    llm = ChatOpenAI(temperature=0, model_name=model)
  print(f"===== model: {model}")

  # chain
  print(f"= INIT CHAIN")
  if memory:
    chain_type_kargs = {'prompt': custom_prompt_history(), 'memory': memory}
  else:
    chain_type_kargs = {'prompt': custom_prompt()}

  chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kargs
  )

  print(f"= START CHAT BOT")
  app.run(debug=True)