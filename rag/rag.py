# import librairies

from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import time
import pandas as pd
import faiss
import matplotlib.pyplot as plt
from typing import Iterator, Tuple, Optional, Union, Callable, Any
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, pipeline, AutoModelForCausalLM, Pipeline
from sentence_transformers import SentenceTransformer
import torch
import numpy as np

# Constants
SEPARATOR = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

def timer(func:Callable[[Any], Any])->Callable[[Any], Any]:
    name=func.__name__
    
    def description(*args, **kwargs):
        arg_str=', '.join(repr(arg) for arg in args)
        start = time.time()
        resultat=func(*args, **kwargs)
        end = time.time()
        if kwargs is None:
            print(f"\nFunction {name}\nargs: {arg_str}\ndone in :{end - start}")
        else:
            key_word = ""
            for key in kwargs.keys():
                if len(repr(kwargs[key])) < 15:
                    key_word += ', ' + key + ": "+ repr(kwargs[key])
                else:
                    key_word += ', ' + key + ": "+ repr(type(kwargs[key]))
            print(f"\nFunction {name}\nargs {arg_str}\nkwargs {key_word}\ndone in :{end - start}")
        return resultat
    return description
class MyParser(BaseBlobParser):
    """A simple parser that creates a document from each line."""

    def lazy_parse(self, blob: Blob) -> Iterator[Document]:
        """Parse a blob into a document line by line."""
        line_number = 0
        with blob.as_bytes_io() as f:
            for line in f:
                line_number += 1
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": blob.source}
                )
@timer
def extract_data(
    path: str = None,
    test_html: bool = False,
    test_csv: bool = False,
)-> list[LangchainDocument]:
    """Load data.
    
    Args:
        path: A string that is a path to the data.
        test_html: A boolean, if True it will load html test data.
        test_csv: A boolean, if True it will load csv test data.

    Returns:
        A list of str containing document information.
        Each object has page_content and metadata attributes. For example:

        print(raw_knowledge_database[0].page_content)

        CET Time
        UK Time (HH:MM)
        Area Code
        Area Name
        Unit Price (inc VAT)        
        
        2021-01-01 00:00:00+00:00
        00:00
        F
        North_Eastern_England
        15.21
        ...

    Raises:
        ValueError: No url or path where given
    """
    if path is not None:
        # file_extension = pathlib.Path(path).suffix
        # print("File Extension: ", file_extension)
        loader = GenericLoader.from_filesystem(path=".", 
                                               glob="*.mdx", 
                                               show_progress=True, 
                                               parser=MyParser(),
                                               )
    
    elif test_html:
        LINK='https://files.energy-stats.uk/csv_output/'
        page = requests.get(LINK)
        soup = BeautifulSoup(page.content, 'html.parser')

        # Create top_items as empty list
        all_links = []

        # Extract and store all links in website
        links = soup.select('a')
        for ahref in links:
            text = ahref.text
            text = text.strip() if text is not None else ''

            href = ahref.get('href')
            href = href.strip() if href is not None else ''
            all_links.append({"href": href, "text": text})

        # Get data from links and convert to csv
        columns = "CET Time,UK Time (HH:MM),Area Code,\
                   Area Name,Unit Price (inc VAT)\n"
        for link in all_links[1:]:
            downloaded_data = urlopen(LINK + link['href'])
            with open("../data/raw_data/" + link['href'], 'w') as file:
                file.write(columns)
                for line in urlopen(LINK + link['href']).readlines():
                    file.write(line.decode())
                file.close()
        glob = "csv_tracker*"
        loader = DirectoryLoader('../data', glob=glob, show_progress=True)

    elif test_csv:
        path = "../data/raw_data/jena_climate_small.csv"
        loader = CSVLoader(file_path=path)

    else:
        raise ValueError("No path where given")

    data = loader.load()
    raw_knowledge_database = [
        LangchainDocument(page_content=doc.page_content, metadata=doc.metadata) for doc in data
    ]
    return raw_knowledge_database

@timer
def split_documents(
    chunk_size: int,
    data_path: str,
    knowledge_base: list[LangchainDocument],
    tokenizer_name: Optional[str] = "thenlper/gte-small",
    plot_path: Optional[str] = None,
    separators: Optional[list[str]] = SEPARATOR
) -> list[str]:
    """Split documents into chunks and return a list of documents.
    
    The function uses a hierarchical list of separators for splitting documents.
    
    Args:
        chunk_size: A integer specifying the maximum chunk size.
        knowledge_base: List of raw data.
        tokenizer_name: A string refering to a tokenizer model.
        plot: A boolean option to display final length distribution.
        separators: A list of separator for chunking documents.

    Returns:
        A list of processed LangchainDocument objects. 
        Each previous object is now divided into chuncks.
    """
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(tokenizer_name),
        chunk_size=chunk_size,
        chunk_overlap=int(chunk_size / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=separators,
    )

    docs_processed = []
    for doc in knowledge_base:
        docs_processed += text_splitter.split_documents([doc])

    # Remove duplicates
    unique_texts = {}
    docs_processed_unique = []
    for doc in docs_processed:
        if doc.page_content not in unique_texts:
            unique_texts[doc.page_content] = True
            docs_processed_unique.append(doc.page_content)
    
    np.save(data_path, docs_processed_unique)

    if plot_path is not None:
        # Let's visualize the chunk sizes
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        lengths = []
        for doc in docs_processed:
            lengths.append(len(tokenizer.encode(doc)))
        fig = pd.Series(lengths).hist()
        plt.title("Document lengths in the knowledge base in tokens")
        plt.savefig(plot_path)
    return docs_processed_unique

@timer
def init_pipeline(
    model_path: Optional[str] = "HuggingFaceH4/zephyr-7b-beta",
    tokenizer_path: Optional[str] = "HuggingFaceH4/zephyr-7b-beta",
    save_path: Optional[str] = None
) -> pipeline:
    """Initialize LLM pipeline
    
    Args:
        model_path: A string that is a model name or a path to a local object.
        tokenizer_path: A string that is a path to a local tokenizer object.
        save_path: A string save path for pipeline model.
    
    Returns:
        A LLM pipeline for text generation.
        A tokenizer adapted to this LLM.
    """
    model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_path, 
            )
    tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path=tokenizer_path
                )

    if save_path is not None:
        model.save_pretrained(save_path)
        tokenizer.save_pretrained(save_path)

    READER_LLM = pipeline(
                 model=model,
                 tokenizer=tokenizer,
                 task="text-generation",
                 do_sample=True,
                 temperature=0.2,
                 repetition_penalty=1.1,
                 return_full_text=False,
                 max_new_tokens=512,
                 )
    return READER_LLM, tokenizer

@timer
def prompt_format(tokenizer: AutoTokenizer) -> Union[list[int], dict]:
    """Apply a prompt template to feed to the Reader LLM.
        
    Args:
        Tokenizer: A tokenizer that will impose a prompt format to the LLM.

    Returns:
        An embeeding of the prompt format.
    """

    prompt_in_chat_format = [
        {
            "role": "system",
            "content": """Using the information contained in the context,
    give a comprehensive answer to the question.
    Respond only to the question asked.
    Response should be concise and relevant to the question.
    Provide the number of the source document when relevant.
    If the answer cannot be deduced from the context, do not give an answer.""",
        },
        {
            "role": "user",
            "content": """Context:
    {context}
    ---
    Now here is the question you need to answer.

    Question: {question}""",
        },
    ]

    return tokenizer.apply_chat_template(
        prompt_in_chat_format, tokenize=False, add_generation_prompt=True
    )

@timer
def answer_with_rag(
    question: str,
    llm: Pipeline,
    data_path: str,
    rag_prompt_format: Union[list[int], dict],
    num_retrieved_docs: int = 30,
) -> Tuple[str, list[str]]:
    """Agregate the whole pipeline, linking processed document, LLM and query.

    Args:
        question: A prompt expressing a user question.
        llm: A LLM pipeline object.
        knowledge_index: A processed document database.
        rag_prompt_format: A prompt format apply to the LLM's response.
        reranker: A reranker that will rank each chunck.
        num_retrieved_docs: A integer specifying max number of retrieved docs.    
    Returns:
        A LLM's answer to a givne query based on a context extracted with RAG.
        The most meaningful documents given a query.

        example:
            TODO ADD EXAMPLE
    """
    # Gather documents with retriever
    start = time.time()
    train = np.load(data_path)
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    docs_encoded = model.encode([doc for doc in train])
    embed_query = np.array([model.encode(question)])
    d = docs_encoded.shape[1]
    if len(train) % 100 > 2:
        nlist = len(train)//100
    else:
        nlist = 10
    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist)
    index.train(train)
    index.add(train)
    distance, index = index.search(np.array([embed_query]), k=num_retrieved_docs)
    relevant_docs = train[index[0][index[0] != -1]]
    end = time.time()
    print(f"Documents retrieved in {end - start}")

    # Build the final prompt
    context = "\nExtracted documents:\n"
    for i, doc in enumerate(relevant_docs):
        context += "".join([f"Document {str(i)}:::\n" + doc])

    final_prompt = rag_prompt_format.format(question=question, context=context)

    # Redact an answer
    start = time.time()
    answer = llm(final_prompt)[0]["generated_text"]
    end = time.time()
    print(f"Answer generated in {end - start}")

    return answer, relevant_docs
