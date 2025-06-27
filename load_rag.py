# rag_core.py

import os
import pickle
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

class SimpleRAG:
    def __init__(self, db_path="rag_db"):

        self.db_path = db_path
        self._load_database()

    def _load_database(self):
        vectorizer_path = os.path.join(self.db_path, r"C:\Users\ASUS\Desktop\graduation project\Rodina Ahmed\final code\rag_db\vectorizer.pkl")
        index_path = os.path.join(self.db_path, r"C:\Users\ASUS\Desktop\graduation project\Rodina Ahmed\final code\rag_db\index.faiss")
        docs_path = os.path.join(self.db_path, r"C:\Users\ASUS\Desktop\graduation project\Rodina Ahmed\final code\rag_db\docs.pkl")

        with open(vectorizer_path, "rb") as f:
            self.vectorizer = pickle.load(f)

        self.index = faiss.read_index(index_path)

        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)

    def search(self, query, top_k=3):
        query_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        _, indices = self.index.search(query_vec, top_k)
        return [self.documents[i]["text"] for i in indices[0] if i < len(self.documents)]