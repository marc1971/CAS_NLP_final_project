
def hello_world(name):
   print('Hallo '+ name)



def load_text(dict_path, N):

    '''
    Imports needed: 
    - os

    Description:
    Loads ths Text files into a list. The function returns a list of N strings.
    
    
    Variables:
    dict_path:      str;  The path of the folder the docs are placed
    N:              int OR str; It is a integer OR the string 'all'. If a integer is found, 
                    the corrsponding number of documents are loaded. If 'all is found, 
                    all documents are loaded.

    Return: list of str
    '''
    import os
    dict_path = dict_path



    entries = os.listdir(dict_path)
    # Filter out entries that are files
    files = [entry for entry in entries if os.path.isfile(os.path.join(dict_path, entry))]
    
    
    
    
    N_docs_all = len(files)


    if isinstance(N, int):
        N_docs = N
        # Handle the integer case
        # For example, if N_docs represents a number of documents to process
        # process_integer_docs(N_docs)
    elif isinstance(N, str):
        N_docs = N_docs_all



    texts =[]
    for i in range (1,N_docs+1):
      with open(dict_path+f'/Doc{i:02}.txt', 'r', encoding = 'utf-8') as file:
        texts.append(file.read())
    
    return texts


#TextSplitter

def text_splitting(chunk_size, overlap, separator=""):
    '''
    Imports needed: 
    - from langchain.text_splitter import CharacterTextSplitter

    Description:
    Splitts the text in a certain number of chunks with an overlap of "overlap".
    The Separator should be st to separator=""!
    The function returns a text_splitter.

    Variabels:
    chunk_size:     int; Numer of Characters in the Chunk (i.e. 500)

    Return: function
    '''
    from langchain.text_splitter import CharacterTextSplitter
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size,
                                     chunk_overlap=overlap,
                                     separator= separator
                                     )
    return text_splitter




# HelperFunction to extract Titel of Document
def get_text_before_newline(text):
   '''
    Imports needed: 
    - re

    Description:
    Helper Function to read the first Line in each Document, which is the title of the document!
    The Regular Expression is to match any character sequence before the first newline character.
    The function returns a srting with the titel of the document.

    Variabels:
    chunk_size:     int; Numer of Characters in the Chunk (i.e. 500)

    Return: str
  
    '''
   import re
   
   match = re.search(r'^(.*?)\n', text, re.DOTALL)
   
   if match:
    return match.group(1)  # Return the matched group, which is the text before the newline
   else:
      return 'No title found!'  
   
# Creating LangChain Document
def create_LangChain_doc(texts, chunk_size, overlap, separator):
   '''
    Imports needed: 
    - from text_loader import get_text_before_newline (Helper Function from above)
    - from text_loder import text_splitting (text_splitter from CharacterTextSplitter is used)

    Description:
    A langchaine-Document with the "create_documents"-methode from TextSplitter is created.
    The TextSplitter uses a list of texts (here: 'texts') and a list of metadatas (here: 'metadatas') to create a list of text chunks
    The helperfunction "get_text_before_newline(text)" is used to read out the title of each document.
    The function returns a langchain-object

    Variabels:
    texts:     list of str; The input variable is a list of strings, with the Titel of the Document in the first line sperated by \n
  
    Return: LangChain-Object
    '''
   from langchain.text_splitter import CharacterTextSplitter
   metadatas = []

   for text in texts:
      metadatas.append({"document": get_text_before_newline(text)})

   documents = text_splitting(chunk_size, overlap, separator).create_documents(texts, metadatas=metadatas)

   return documents


#Chunk the LangChain Object
def chunk_docs(documents,chunk_size, overlap, separator):
    '''
    Imports needed: 
    - from text_loder import text_splitting (text_splitter from CharacterTextSplitter sis used)

    Description:
    The LangChain-Object is split up into the nummer of Chunks, that is defined in the text_splitting function and stored in a List of LangChain-Objects. 
    The function returns a List of LangChain-Objects.

    Variabels:
    documents:  LangChain-Object; The input variable is a LangChain-Object
  
    Return:     List of LangChain-Objects
    '''
    from langchain.text_splitter import CharacterTextSplitter
    chunked_documents = text_splitting(chunk_size, overlap, separator).split_documents(documents)
    return chunked_documents