import argparse
import sys
import os
sys.path.append(os.path.abspath("./rag/"))
from rag import extract_data, split_documents

def main():
    "Creation of the data command"
    parser = argparse.ArgumentParser(
        description="Create and save FAISS object given raw data")
    parser.add_argument('--extract_path', 
                        metavar='extract_path', 
                        type=str,
                        default=None,
                        help="Raw data path")
    parser.add_argument('--test_html', 
                        metavar='test_html', 
                        type=bool,
                        default=False,
                        help="Option for test1 load")
    parser.add_argument('--test_csv', 
                        metavar='test_csv', 
                        type=bool,
                        default=False,
                        help="Option for test2 load")
    parser.add_argument('--chunk_size', 
                        metavar='chunk_size', 
                        type=int,
                        default=512,
                        help="Maximum size of each chunk")
    parser.add_argument('--data_path', 
                        metavar='data_path', 
                        type=str,
                        default=None,
                        help="Save path for chuncked data")
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

    args = parser.parse_args()

    chunk_size = args.chunk_size
    tokenizer_name = args.tokenizer_name
    extract_path = args.extract_path
    test_html = args.test_html
    test_csv = args.test_csv
    data_path = args.data_path
    plot_path = args.plot_path
    separators = args.separators

    raw_knowledge_base = extract_data(
                         path=extract_path,
                         test_html=test_html,
                         test_csv=test_csv,
                         )

    docs_processed = split_documents(
                     chunk_size=chunk_size,
                     data_path=data_path,
                     knowledge_base=raw_knowledge_base,
                     tokenizer_name=tokenizer_name,
                     plot_path=plot_path,
                     separators=separators
                     )

if __name__=="__main__":
    main()