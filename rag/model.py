import argparse
import sys
import os
sys.path.append(os.path.abspath("./rag/"))
from rag import init_pipeline, prompt_format, answer_with_rag

def main():
    """Creation of the model command."""
    parser = argparse.ArgumentParser(
        description="Output an LLM's answer to a question on documents")
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
    parser.add_argument('--question', 
                        metavar='question', 
                        type=str,
                        default="What is the temperature evolution on the 26 December of 2016?",
                        help="question on data for llm")
    parser.add_argument('--data_path', 
                        metavar='data_path', 
                        type=str,
                        default=None,
                        help="data path for context")
    parser.add_argument('--num_retrieved_docs', 
                        metavar='num_retrieved_docs', 
                        type=int,
                        default=24,
                        help="Max number of retrieved docs")

    args = parser.parse_args()

    model_path = args.model_path
    tokenizer_path = args.tokenizer_path
    save_path = args.save_path
    tokenizer_path = args.tokenizer_path
    question = args.question
    data_path = args.data_path
    num_retrieved_docs = args.num_retrieved_docs

    reader_llm, tokenizer = init_pipeline(model_path=model_path,
                               tokenizer_path=tokenizer_path,
                               save_path=save_path)

    rag_prompt_format = prompt_format(tokenizer=tokenizer)

    # reranker = init_reranker(name=reranker_name)

    output = answer_with_rag(question=question,
                             llm=reader_llm,
                             data_path=data_path,
                             rag_prompt_format=rag_prompt_format,
                             num_retrieved_docs=num_retrieved_docs,
                             )

    print(output)

if __name__=="__main__":
    main()