from typing import List
from fashion_recommender.sink import Vector
from fashion_recommender.sink.clients.milvus import MilvusClient


class VectorSink:
    def __init__(self, client: MilvusClient):
        self._client = client

    def sink(self, identifiers: List, data: List[List], structure: str):
        vectors = [Vector(name=idx, vector=d) for idx, d in zip(identifiers, data)]
        self._client.insert_vectors(structure=structure, vectors=vectors)

