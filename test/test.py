import unittest
import sys
import os
import json

sys.path.append(os.path.abspath("../rag/"))

from rag import *

# Load the JSON data from your file
# FIXME
with open('test1.json') as json_file:
    test_args = json.load(json_file)['testParameters']

# Describe test
print("\nTest with following parameters :\n")
for key in test_args.keys():
    print(f"{key} : {test_args[key]}\n")
class TestRag(unittest.TestCase):
    def test_rag(self):
        print("\nTest extract function")
        doc = extract_data(
              path=test_args["extract_path"],
              test_html=test_args["test_html"],
              test_csv=test_args["test_csv"],
              )
        self.assertTrue(doc is not None)
        print("Passed")
        print("\nTest split function")
        docs_processed_unique = split_documents(
                                chunk_size=test_args["chunk_size"],
                                knowledge_base=doc,
                                tokenizer_name=test_args["tokenizer_name"],
                                plot=test_args["plot"],
                                separators=test_args["separators"],
                                )
        self.assertTrue(docs_processed_unique is not None)
        print("Passed")
        print("\nTest init_emmbedding function")
        embedding_model = init_embedding_model(
                          embedding_model_name=test_args["embedding_model_name"],
                          multiprocess=test_args["multiprocess"],
                          model_kwargs=test_args["model_kwargs"],
                          encode_kwargs=test_args["encode_kwargs"],
                          )
        self.assertTrue(embedding_model is not None)
        print("Passed")
        print("\nTest create faiss function")
        knowledge_database = create_faiss(
                             embedding_model=embedding_model,
                             docs_processed=docs_processed_unique,
                             save_path=test_args["faiss_save_path"],
                             )
        self.assertTrue(knowledge_database is not None)
        print("Passed")
        print("\nTest load faiss function")
        faiss = load_faiss(
                path=test_args["load_faiss_path"],
                embedding_model=embedding_model,
                )
        self.assertTrue(faiss is not None)
        print("Passed")
        print("\nTest init pipeline function")
        llm = init_pipeline(
              model_path=test_args["model_path"],
              tokenizer_path=test_args["tokenizer_path"],
              save_path=test_args["llm_save_path"],
              )
        self.assertTrue(llm is not None)
        print("Passed")
        print("\nTest prompt format function")
        rag_prompt_format = prompt_format(tokenizer=embedding_model)
        self.assertTrue(rag_prompt_format is not None)
        print("Passed")
        print("\nTest answer with rag function")
        output = answer_with_rag(
        question=test_args["question"],
        llm=llm,
        knowledge_index=knowledge_database,
        rag_prompt_format=rag_prompt_format,
        num_retrieved_docs=test_args["num_retrieved_docs"],
        num_docs_final=test_args["num_docs_final"],
        )
        self.assertTrue(output is not None)
        print('\nOutput of the test: \n' + output)

if __name__ == '__main__':
    unittest.main()