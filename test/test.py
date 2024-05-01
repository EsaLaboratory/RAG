import unittest
import sys
import os
import json
import time

sys.path.append(os.path.abspath("../rag/"))

from rag import *
class TestRag(unittest.TestCase):
    FILENAME = "1"

    def test_rag(self):
        with open('test_params.json') as json_file:
            file = 'test' + str(self.FILENAME)
            args = json.load(json_file)["testParameters"][file]

        # Describe test
        print("\nTest with following parameters :\n")
        for key in args.keys():
            print(f"{key} : {args[key]}\n")
        
        doc = extract_data(
              path=args["extract_path"],
              test_html=args["test_html"],
              test_csv=args["test_csv"],
              )
        self.assertTrue(doc is not None)

        docs_processed_unique = split_documents(
                                chunk_size=args["chunk_size"],
                                knowledge_base=doc,
                                tokenizer_name=args["tokenizer_name"],
                                plot_path=args["plot_path"],
                                separators=args["separators"],
                                )
        self.assertTrue(docs_processed_unique is not None)

        embedding_model = init_embedding_model(
                          embedding_model_name=args["embedding_model_name"],
                          multiprocess=args["multiprocess"],
                          model_kwargs=args["model_kwargs"],
                          encode_kwargs=args["encode_kwargs"],
                          )
        self.assertTrue(embedding_model is not None)

        knowledge_database = create_faiss(
                             embedding_model=embedding_model,
                             docs_processed=docs_processed_unique,
                             save_path=args["faiss_save_path"],
                             )
        self.assertTrue(knowledge_database is not None)

        faiss = load_faiss(
                path=args["load_faiss_path"],
                embedding_model=embedding_model,
                )
        self.assertTrue(faiss is not None)

        llm = init_pipeline(
              model_path=args["model_path"],
              tokenizer_path=args["tokenizer_path"],
              save_path=args["llm_save_path"],
              )
        self.assertTrue(llm is not None)

        rag_prompt_format = prompt_format(tokenizer=embedding_model)
        self.assertTrue(rag_prompt_format is not None)

        output = answer_with_rag(
        question=args["question"],
        llm=llm,
        knowledge_index=knowledge_database,
        rag_prompt_format=rag_prompt_format,
        num_retrieved_docs=args["num_retrieved_docs"],
        num_docs_final=args["num_docs_final"],
        )
        self.assertTrue(output is not None)
        print('\nOutput of the test: \n' + output)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        TestRag.FILENAME = sys.argv.pop()
    unittest.main()
