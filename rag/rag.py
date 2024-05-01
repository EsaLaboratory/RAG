# import librairies

from bs4 import BeautifulSoup
from urllib.request import urlopen
import requests
import time
import pandas as pd
import matplotlib.pyplot as plt
from typing import Iterator, Tuple, Optional, Union, Callable, Any
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import DirectoryLoader
from langchain_core.document_loaders import BaseBlobParser, Blob
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, BitsAndBytesConfig, pipeline, AutoModelForCausalLM, Pipeline
import torch

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
            key_word=', '.join(key + ": "+ repr(kwargs[key]) for key in kwargs.keys())
            print(f"\nFunction {name}\nargs {arg_str}\nkwargs {key_word}\ndone in :{end - start}")
        return resultat
    return description

READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
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
    test_html: bool = True,
    test_csv: bool = False,
)-> list[LangchainDocument]:
    """Load data.
    
    Args:
        path: A string that is a path to the data.
        test_html: A boolean, if True it will load html test data.
        test_csv: A boolean, if True it will load csv test data.

    Returns:
        A list of LangchainDocument objects containing document information.
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
        path = "../data/raw_data/jena_climate_2009_2016.csv"
        loader = CSVLoader(file_path=path)

    else:
        raise ValueError("No path where given")

    # Storing data into langchain format
    data = loader.load()
    raw_knowledge_database = [
        LangchainDocument(page_content=doc.page_content, metadata=doc.metadata)
        for doc in data
    ]
    return raw_knowledge_database

@timer
def split_documents(
    chunk_size: int,
    knowledge_base: list[LangchainDocument],
    tokenizer_name: Optional[str] = "thenlper/gte-small",
    plot_path: Optional[str] = None,
    separators: Optional[list[str]] = SEPARATOR
) -> list[LangchainDocument]:
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
            docs_processed_unique.append(doc)

    if plot_path is not None:
        # Let's visualize the chunk sizes
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        lengths = []
        for doc in docs_processed:
            lengths.append(len(tokenizer.encode(doc.page_content)))
        fig = pd.Series(lengths).hist()
        plt.title("Document lengths in the knowledge base in tokens")
        plt.savefig(plot_path)
    return docs_processed_unique

@timer
def init_embedding_model(
    embedding_model_name: Optional[str] = "thenlper/gte-small",
    multiprocess: Optional[bool] = True,
    model_kwargs: Optional[dict] = {"device": "cpu"}, # gpu 
    encode_kwargs: Optional[dict] = {"normalize_embeddings": True},
    save_path: Optional[str] = None,
) -> HuggingFaceEmbeddings:
    """Initialize an embedding model.
    
    Args:
        embedding_model_name: A string that refers to an embedding model.
        multiprocess: A boolean that defines multiprocessing parameter.
        model_kwargs: A dict containing device settings.
        encode_kwargs: A dict containing encoding settings.
        save_path: A string that is a save path for the loaded embedding model.
        
    Returns:
        An embedding model that will convert text into tokens.
    """
    model_kwargs['quantization_config'] = BitsAndBytesConfig(
                 load_in_4bit=True,
                 bnb_4bit_use_double_quant=True,
                 bnb_4bit_quant_type="nf4",
                 bnb_4bit_compute_dtype=torch.bfloat16,
                 )
    embedding_model = HuggingFaceEmbeddings(
                      model_name=embedding_model_name,
                      multi_process=multiprocess,
                      model_kwargs=model_kwargs,
                      encode_kwargs=encode_kwargs,
                      cache_folder=save_path,
                      )
    return embedding_model

@timer
def create_faiss(
    embedding_model: HuggingFaceEmbeddings,
    docs_processed: list[LangchainDocument],
    save_path: str
) -> FAISS:
    """Create and save the retrieving database.
    
    Args:
        embedding_model: An embedding model adapted the processed documents.
        docs_processed: List of processed chunked data.
        save_path: A string refering to a saving path.

    Returns:
        A FAISS object assimilited as a database that we will query.
    """
    KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
                                docs_processed,
                                embedding_model,
                                distance_strategy=DistanceStrategy.COSINE,
                                )
    KNOWLEDGE_VECTOR_DATABASE.save_local(save_path)
    return KNOWLEDGE_VECTOR_DATABASE

@timer
def load_faiss(
    path: Union[str, list[str]],
    embedding_model: HuggingFaceEmbeddings
) -> FAISS:
    """Load faiss object.

    When several path are given, it merges all faiss object into one.

    Args:
        path: A string or list of string refering to a local FAISS object.
        embedding_model: A string refering to a LLM name.
    
    Returns:
        A FAISS object assimilited as a database that we will query.
    """
    if type(path) == list:
        faiss = FAISS.load_local(path[0], embedding_model)
        for path_to_faiss in path[1:]:
            faiss.merge_from(FAISS.load_local(path_to_faiss, embedding_model))
    else:
        faiss = FAISS.load_local(path, embedding_model)
    return faiss

@timer
def init_pipeline(
    model_path: Optional[str] = None,
    tokenizer_path: Optional[str] = None,
    save_path: Optional[str] = None
) -> pipeline:
    """Initialize LLM pipeline
    
    Args:
        model_path: A string that is a model name or a path to a local object.
        tokenizer_path: A string that is a path to a local tokenizer object.
        save_path: A string save path for pipeline model.
    
    Returns:
        A LLM pipeline for text generation.
    """
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if model_path is not None and tokenizer_path is not None:
        model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_path, 
                quantization_config=bnb_config
                )
        tokenizer = AutoTokenizer.from_pretrained(
                    pretrained_model_name_or_path=tokenizer_path
                    )
    else:
        model = AutoModelForCausalLM.from_pretrained(
                READER_MODEL_NAME,
                quantization_config=bnb_config
                )
        tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
    
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
        max_new_tokens=500,
    )
    return READER_LLM

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



# def init_reranker(name: Optional[str] = "colbert-ir/colbertv2.0"
#     ) -> RAGPretrainedModel:
#     """Initialize a reranker.
    
#     A reranker allow to retrieve more documents and to order them well.
#     Colbertv2 computes good interactions between query and documents.

#     Args:
#         name: A string that refers to reranker model.

#     Returns:
#         A reranker object. 
#     """
#     return RAGPretrainedModel.from_pretrained(name)

@timer
def answer_with_rag(
    question: str,
    llm: Pipeline,
    knowledge_index: FAISS,
    rag_prompt_format: Union[list[int], dict],
    # reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 24,
) -> Tuple[str, list[LangchainDocument]]:
    """Agregate the whole pipeline, linking processed document, LLM and query.

    Args:
        question: A prompt expressing a user question.
        llm: A LLM pipeline object.
        knowledge_index: A processed document database.
        rag_prompt_format: A prompt format apply to the LLM's response.
        reranker: A reranker that will rank each chunck.
        num_retrieved_docs: A integer specifying max number of retrieved docs.
        num_docs_final: A integer specifying max number of context docs.
    
    Returns:
        A LLM's answer to a givne query based on a context extracted with RAG.
        The most meaningful documents given a query.

        example:
            TODO ADD EXAMPLE
    """
    
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(
                    query=question, 
                    k=num_retrieved_docs
                    )
    relevant_docs = [doc.page_content for doc in relevant_docs]

    # Optionally rerank results
    # if reranker:
    #     print("=> Reranking documents...")
    #     relevant_docs = reranker.rerank(
    #                     question, 
    #                     relevant_docs, 
    #                     k=num_docs_final
    #                     )
    #     relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    for i, doc in enumerate(relevant_docs):
        context += "".join([f"Document {str(i)}:::\n" + doc])

    final_prompt = rag_prompt_format.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs
