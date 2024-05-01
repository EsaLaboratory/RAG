import argparse
import sys
import os
sys.path.append(os.path.abspath("./rag/"))
from rag import extract_data, split_documents, create_faiss

def main():
    "Creation of the data command"
    parser = argparse.ArgumentParser(
        description="Create and save FAISS object given raw data")
    parser.add_argument('--chunk_size', 
                        metavar='chunk_size', 
                        type=int,
                        default=512,
                        help="Maximum size of each chunk")
    parser.add_argument('--tokenizer_name', 
                        metavar='tokenizer_name', 
                        type=str,
                        default="thenlper/gte-small",
                        help="Name of tokenizer model")
    parser.add_argument('--plot_path', 
                        metavar='plot_path', 
                        type=str,
                        default=None,
                        help="Save path for plot of chunck size distribution")
    parser.add_argument('--separators', 
                        metavar='separators', 
                        type=str,
                        default=None,
                        help="Define list of separators")
    parser.add_argument('--embedding_model', 
                        metavar='embedding_model', 
                        type=str,
                        default="thenlper/gte-small",
                        help="Name of embedding model")
    parser.add_argument('--save_path',
                        metavar='save_path', 
                        type=str,
                        default=None,
                        help="path of the local tokenizer (optional)")

    args = parser.parse_args()

    chunk_size = args.chunk_size
    tokenizer_name = args.tokenizer_name
    plot_path = args.plot_path
    separators = args.separators
    embedding_model = args.embedding_model
    save_path = args.save_path

    raw_knowledge_base = extract_data(url=None,
                                      path=None)
    
    docs_processed = split_documents(chunk_size=chunk_size,
                                            knowledge_base=raw_knowledge_base,
                                            tokenizer_name=tokenizer_name,
                                            plot_path=plot_path,
                                            separators=separators)

    knowledge_vector_database = create_faiss(embedding_model=embedding_model,
                                             docs_processed=docs_processed,
                                             save_path=save_path)

if __name__=="__main__":
    main()