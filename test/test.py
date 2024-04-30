import unittest
import sys
import argparse
import json

sys.path.insert(1, '../rag/rag.py')

import rag

parser = argparse.ArgumentParser()
parser.add_argument('-ep', '--extractpath', help='extract path variable')
parser.add_argument('-h', '--html', help='html variable')
parser.add_argument('-c', '--csv', help='csv variable')

# Load the JSON data from your file (replace 'your_file.json' with the actual filename)
with open('your_file.json') as json_file:
    json_data = json.load(json_file)

# Extract the values from the JSON dictionary
args_list = []
for key in json_data['testParameters']['test1']:
    args_list.append(key)
    args_list.append(json_data['testParameters']['test1'][key])
args = parser.parse_args(args_list)

# Describe test
print("\nTest with following parameters :\n")
for key in args.key:
    print(f"{key} : {args.key}\n")
class TestRag(unittest.TestCase):
    def __init__(self):
        self.doc = None
        self.docs_processed_unique = None
        self.embedding_model = None
        self.knowledge_database = None
        self.faiss = None
        self.llm = None
        self.rag_prompt_format = None
        self.output = None

    def test_extract(self):
        doc = rag.extract_data(
                path=args.path,
                test_html=args.test_html,
                test_csv=args.test_csv
                )
        self.doc = doc
        self.assertTrue(doc is not None)

    def test_split(self):
        docs_processed_unique = rag.split_documents(
                                    chunk_size=args.chunk_size,
                                    knowledge_base=self.doc,
                                    tokenizer_name=args.tokenizer_name,
                                    plot=args.plot,
                                    separators=args.separators
                                    )
        self.docs_processed_unique = docs_processed_unique
        self.assertTrue(docs_processed_unique is not None)

    def test_init_embedding(self):
        embedding_model = rag.init_embedding_model(
            embedding_model_name=args.embedding_model_name,
            multiprocess=args.multiprocess,
            model_kwargs=args.model_kwargs,
            encode_kwargs=args.encode_kwargs
            )
        self.embedding_model = embedding_model
        self.assertTrue(embedding_model is not None)
    
    def test_init_faiss(self, save_path):
        knowledge_database = rag.create_faiss(
            embedding_model=self.embedding_model,
            docs_processed=self.docs_processed_unique,
            save_path=args.faiss_save_path
        )
        self.knowledge_database = knowledge_database
        self.assertTrue(knowledge_database is not None)

    
    def test_load_faiss(self):
        faiss = rag.load_faiss(
            path=args.load_faiss_path,
            embedding_model=self.embedding_model
        )
        self.faiss = faiss
        self.assertTrue(faiss is not None)

    def test_init_pipeline(self):
        llm = rag.init_pipeline(
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            save_path=args.llm_save_path
        )
        self.llm = llm
        self.assertTrue(llm is not None)

    def test_prompt_format(self):
        rag_prompt_format = rag.prompt_format(tokenizer=self.embedding_model)
        self.rag_prompt_format = rag_prompt_format
        self.assertTrue(rag_prompt_format is not None)
    
    def test_answer(self):
        output = rag.answer_with_rag(
        question=args.question,
        llm=self.pipeline,
        knowledge_index=self.knowledge_database,
        rag_prompt_format=self.rag_prompt_format,
        num_retrieved_docs=args.num_retrieved_docs,
        num_docs_final=args.num_docs_final
        )
        self.output = output
        self.assertTrue(output is not None)

if __name__ == '__main__':
    unittest.main()