import argparse
import sys
import os
sys.path.append(os.path.abspath("./rag/"))
from rag import answer_with_rag, init_pipeline, load_faiss, prompt_format, init_embedding_model

def main():
    """Creation of the model command."""
    parser = argparse.ArgumentParser(
        description="Output an LLM's answer to a question on documents")
    # parser.add_argument('--reranker', 
    #                     metavar='reranker',
    #                     type=str,
    #                     help="Computes interactions between query document")
    parser.add_argument('--question', 
                        metavar='question', 
                        type=str,
                        default="What is the temperature evolution in Kelvin?",
                        help="question on data for llm")
    parser.add_argument('--model_path', 
                        metavar='model_path', 
                        type=str,
                        default="HuggingFaceH4/zephyr-7b-beta",
                        help="path of the local model (optional)")
    parser.add_argument('--tokenizer_path', 
                        metavar='tokenizer_path', 
                        type=str,
                        default="HuggingFaceH4/zephyr-7b-beta",
                        help="path of the local tokenizer (optional)")
    parser.add_argument('--save_path', 
                        metavar='save_path', 
                        type=str,
                        default=None,
                        help="save path for llm and tokenizer (optional)")
    parser.add_argument('--embedding_model_name', 
                        metavar='embedding_model_name', 
                        type=str,
                        default="thenlper/gte-small",
                        help="Name of embedding model (optional)")
    parser.add_argument('--multiprocess', 
                        metavar='multiprocess', 
                        type=bool,
                        default=True,
                        help="Options loading embbeding (optional)")
    parser.add_argument('--model_kwargs', 
                        metavar='model_kwargs', 
                        type=dict,
                        default={"device": "cpu"},
                        help="Embeding kwargs, format json (optional)")
    parser.add_argument('--encode_kwargs',
                        metavar='encode_kwargs', 
                        type=str,
                        default={"normalize_embeddings": True},
                        help="Embeding kwargs, format json (optional)")
    parser.add_argument('--faiss_path', 
                        metavar='faiss_path', 
                        type=str,
                        default="../data/faiss/test1/faiss_index",
                        help="Path for local faiss object")

    args = parser.parse_args()

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    save_path = args.save_path
    embedding_name = args.embedding_model_name
    multiprocess = True if args.multiprocess is not None else False
    model_kwargs = args.model_kwargs
    encode_kwargs = args.encode_kwargs
    if args.faiss_path is not None:
        faiss_path = args.faiss_path
    else:
        raise Exception("Provide path to faiss object with the option --faiss")
    tokenizer_path = args.tokenizer_path
    # reranker_name = args.reranker
    question = args.question

    reader_llm = init_pipeline(model_path=model_path,
                               tokenizer_path=tokenizer_path,
                               save_path=save_path)

    embedding_model = init_embedding_model(embedding_model_name=embedding_name,
                                           multiprocess=multiprocess,
                                           model_kwargs=model_kwargs, 
                                           encode_kwargs=encode_kwargs)

    knowledge_database = load_faiss(path=faiss_path,
                                    embedding_model=embedding_model)

    rag_prompt_format = prompt_format(tokenizer=embedding_model)

    # reranker = init_reranker(name=reranker_name)

    output = answer_with_rag(question=question,
                             llm=reader_llm,
                             knowledge_index=knowledge_database,
                             rag_prompt_format=rag_prompt_format,
                            #  reranker=reranker,
                             num_retrieved_docs=num_retrieved_docs,
                             num_docs_final=num_docs_final)

    print(output)

if __name__=="__main__":
    main()