import unittest
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
sys.path.insert(1, '../rag/rag.py')

import rag

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

    def test_extract(self, path, test_html, test_csv):
        doc = rag.extract_data(
                path=path,
                test_html=test_html,
                test_csv=test_csv
                )
        self.doc = doc
        self.assertTrue(doc is not None)

    def test_split(self, chunk_size, tokenizer_name, plot, separators):
        docs_processed_unique = rag.split_documents(
                                    chunk_size=chunk_size,
                                    knowledge_base=self.doc,
                                    tokenizer_name=tokenizer_name,
                                    plot=plot,
                                    separators=separators
                                    )
        self.docs_processed_unique = docs_processed_unique
        self.assertTrue(docs_processed_unique is not None)

    def test_init_embedding(self, embedding_model_name, multiprocess, model_kwargs, encode_kwargs):
        embedding_model = rag.init_embedding_model(
            embedding_model_name=embedding_model_name,
            multiprocess=multiprocess,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
            )
        self.embedding_model = embedding_model
        self.assertTrue(embedding_model is not None)
    
    def test_init_faiss(self, save_path):
        knowledge_database = rag.create_faiss(
            embedding_model=self.embedding_model,
            docs_processed=self.docs_processed_unique,
            save_path=save_path
        )
        self.knowledge_database = knowledge_database
        self.assertTrue(knowledge_database is not None)

    
    def test_load_faiss(self, path):
        faiss = rag.load_faiss(
            path=path,
            embedding_model=self.embedding_model
        )
        self.faiss = faiss
        self.assertTrue(faiss is not None)

    def test_init_pipeline(self, model_path, tokenizer_path, save_path):
        llm = rag.init_pipeline(
            model_path=model_path,
            tokenizer_path=tokenizer_path,
            save_path=save_path
        )
        self.llm = llm
        self.assertTrue(llm is not None)

    def test_prompt_format(self):
        rag_prompt_format = rag.prompt_format(tokenizer=self.embedding_model)
        self.rag_prompt_format = rag_prompt_format
        self.assertTrue(rag_prompt_format is not None)
    
    def test_answer(self, question, num_retrieved_docs, num_docs_final):
        output = rag.answer_with_rag(
        question=question,
        llm=self.pipeline,
        knowledge_index=self.knowledge_database,
        rag_prompt_format=self.rag_prompt_format,
        num_retrieved_docs=num_retrieved_docs,
        num_docs_final=num_docs_final
        )
        self.output = output
        self.assertTrue(output is not None)

if __name__ == '__main__':
    unittest.main()