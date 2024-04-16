
###################
#Set up Quantization of Model
###################

# Activate 4-bit precision base model loading
use_4bit = True

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "float16"

# Quantization type (fp4 or nf4)
bnb_4bit_quant_type = "nf4"

# Activate nested quantization for 4-bit base models (double quantization)
use_nested_quant = False

compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=use_4bit,
    bnb_4bit_quant_type=bnb_4bit_quant_type,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=use_nested_quant,
    )

##########################
# Load the model of choice
##########################

def load_llm():
  model_name = "mistralai/Mistral-7B-Instruct-v0.2"

  tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
  tokenizer.pad_token = tokenizer.eos_token
  tokenizer.padding_side = "right"

  model = AutoModelForCausalLM.from_pretrained(
      model_name,
      quantization_config = bnb_config,
      trust_remote_code = True
      )

  text_generation_pipeline = pipeline(
                            model=model,
                            tokenizer=tokenizer,
                            task="text-generation",
                            temperature=0.2,
                            repetition_penalty=1.1,
                            return_full_text=True,
                            max_new_tokens=500)

  llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

  return llm


#####################
#Set Up the RAG
#####################

def get_rag_chain(llm = st.session_state["llm"]):

# Load Database

  embeddings = HuggingFaceEmbeddings(model_name='PM-AI/bi-encoder_msmarco_bert-base_german') #'sentence-transformers/all-mpnet-base-v2'
  db = FAISS.load_local('C:/Users/M1HB/OneDrive - PHBern/Dokumente/PHBern/persönlich/Weiterbildungen/CAS NLP/Final Project/final_project/documents_index', embeddings)

  # '/content/drive/My Drive/ColabNotebooks/CAS_NLP_final_project/streamlit/documents_index'

  context_prompt_template = """
    ### [INST]

    Instruction: Beantworte die Frage aufgrund deines Wissens über die PHBern. 
    Antworte in **ZWEI bis DREI DEUTSCHEN Sätzen**! 
    Wenn du die Antwort nicht findest, dass sage, dass du in den Dokumenten keine Antwort finden kannst.
    Wenn du keine Antwort finden kannst, dann verweise an die Studienberatung der PHBern
    Sei konzise und beantworte nur gestellte Fragen!
    
    Hier ist der Kontext als Hilfe:

    {context}

    ### QUESTION:
    {question}

    [/INST]
    """


    # Here is the chat history until now. Chat History: {chat_history}
  context_prompt = PromptTemplate(input_variables=["context", "question"], template = context_prompt_template)

    # "chat_history"

  llm_chain = LLMChain(llm=llm, prompt=context_prompt)
  retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
    )
  rag_chain = ( {"context": retriever, "question": RunnablePassthrough()} | llm_chain)

  return rag_chain

######################
#Set Up Streamlit App
######################


# Set the title for the Streamlit app

st.title("ChatBot PHBern")
st.image('C:/Users/M1HB/OneDrive - PHBern/Dokumente/PHBern/persönlich/Weiterbildungen/CAS NLP/Final Project/final_project')

# /content/drive/My Drive/ColabNotebooks/CAS_NLP_final_project/streamlit/PHBern.png

# Load the language model

if 'llm' not in st.session_state:
  st.session_state['llm'] = load_llm()

############################
#Set Up Conversational Chat
##########################

def conversational_chat(question):
  start_time = time.time()
  print(question)

#chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=db.as_retriever())
  rag_chain = st.session_state["rag_chain"]
  
  response = rag_chain.invoke(question)

  print("\n1", response)

#response= chain({"question": question}) #"chat_history": st.session_state['history']

  result = response.get("text")
  st.session_state['history'].append((question, result))

#result["answer"]
  
  print("\n", response)

  time_taken = round(time.time()- start_time)
  time_taken_str = "\n\n[ Query executed in "
  
  if time_taken >61:
    mins = time_taken/60
    secs = time_taken-mins*60
    time_taken_str = time_taken_str + str(round(mins)) + " minutes and " + str(round(secs)) + " seconds.]"

  else:
    time_taken_str = time_taken_str + str(time_taken) + " seconds.]"

  result = result + time_taken_str

  return result

###################
#Initialize History and Massages
###################

if "rag_chain" not in st.session_state:
  st.session_state["rag_chain"] = get_rag_chain(st.session_state["llm"])

if 'history' not in st.session_state:
  st.session_state['history'] = []


# Initialize messages

if 'generated' not in st.session_state:
  st.session_state['generated'] = ["Hallo, ich bin der PHBern Bot! (Mistral-7B-Instruct-v0.2). Was möchtest Du über die PHBern wissen?"]

if 'past' not in st.session_state:
  st.session_state['past'] = ["Hey ! 👋"]

  ###############
  # Create containers for chat history and user input
  ###############


# Create containers for chat history and user input

response_container = st.container()
container = st.container()

# User input form

with container:
  with st.form(key='my_form', clear_on_submit=True):
    user_input = st.text_input("question:", placeholder="Ask me anything. 👉 (:", key='input')

submit_button = st.form_submit_button(label='Send')


if submit_button and user_input:
  output = conversational_chat(question=user_input)
  print(output)

st.session_state['past'].append(user_input)
st.session_state['generated'].append(output)
