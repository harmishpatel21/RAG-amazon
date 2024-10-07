import faiss

class Retriever:
    def __init__(self, embeddings, texts):
        self.index = faiss.IndexFlatL2(embeddings[0].shape[1])
        self.index.add(embeddings)
        self.texts = texts

    def retrieve(self, query_embedding, k=5):
        distances, indices = self.index.search(query_embedding, k)
        return [self.texts[i] for i in indices[0]]



