import unittest
import sys
import os
import argparse
import json

sys.path.append(os.path.abspath("../rag/"))

from rag import *

parser = argparse.ArgumentParser()
parser.add_argument('--test', help='test params file')
args = parser.parse_args()
# Load the JSON data from your file (replace 'your_file.json' with the actual filename)
with open('test' + args.test + '.json') as json_file:
    test_args = json.load(json_file)['testParameters']

# Describe test
print("\nTest with following parameters :\n")
for key in test_args.keys():
    print(f"{key} : {test_args[key]}\n")
# class TestRag(unittest.TestCase):
#     def __init__(self):
#         self.doc = None
#         self.docs_processed_unique = None
#         self.embedding_model = None
#         self.knowledge_database = None
#         self.faiss = None
#         self.llm = None
#         self.rag_prompt_format = None
#         self.output = None

#     def test_extract(self):
#         doc = extract_data(
#                 path=test_args.path,
#                 test_html=test_args.test_html,
#                 test_csv=test_args.test_csv
#                 )
#         self.doc = doc
#         self.assertTrue(doc is not None)

#     def test_split(self):
#         docs_processed_unique = split_documents(
#                                     chunk_size=test_args.chunk_size,
#                                     knowledge_base=self.doc,
#                                     tokenizer_name=test_args.tokenizer_name,
#                                     plot=test_args.plot,
#                                     separators=test_args.separators
#                                     )
#         self.docs_processed_unique = docs_processed_unique
#         self.assertTrue(docs_processed_unique is not None)

#     def test_init_embedding(self):
#         embedding_model = init_embedding_model(
#             embedding_model_name=test_args.embedding_model_name,
#             multiprocess=test_args.multiprocess,
#             model_kwargs=test_args.model_kwargs,
#             encode_kwargs=test_args.encode_kwargs
#             )
#         self.embedding_model = embedding_model
#         self.assertTrue(embedding_model is not None)
    
#     def test_init_faiss(self, save_path):
#         knowledge_database = create_faiss(
#             embedding_model=self.embedding_model,
#             docs_processed=self.docs_processed_unique,
#             save_path=test_args.faiss_save_path
#         )
#         self.knowledge_database = knowledge_database
#         self.assertTrue(knowledge_database is not None)

    
#     def test_load_faiss(self):
#         faiss = load_faiss(
#             path=test_args.load_faiss_path,
#             embedding_model=self.embedding_model
#         )
#         self.faiss = faiss
#         self.assertTrue(faiss is not None)

#     def test_init_pipeline(self):
#         llm = init_pipeline(
#             model_path=test_args.model_path,
#             tokenizer_path=test_args.tokenizer_path,
#             save_path=test_args.llm_save_path
#         )
#         self.llm = llm
#         self.assertTrue(llm is not None)

#     def test_prompt_format(self):
#         rag_prompt_format = prompt_format(tokenizer=self.embedding_model)
#         self.rag_prompt_format = rag_prompt_format
#         self.assertTrue(rag_prompt_format is not None)
    
#     def test_answer(self):
#         output = answer_with_rag(
#         question=test_args.question,
#         llm=self.pipeline,
#         knowledge_index=self.knowledge_database,
#         rag_prompt_format=self.rag_prompt_format,
#         num_retrieved_docs=test_args.num_retrieved_docs,
#         num_docs_final=test_args.num_docs_final
#         )
#         self.output = output
#         self.assertTrue(output is not None)

# if __name__ == '__main__':
#     unittest.main()