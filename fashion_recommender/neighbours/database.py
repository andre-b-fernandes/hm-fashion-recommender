from fashion_recommender.clients.milvus import MilvusClient
from fashion_recommender.sink import Vector

from typing import List

class DatabaseNeighbours:
    def __init__(self, client: MilvusClient) -> None:
        self._client = client

    def search(self, vectors: List[List[float]]):
        pass

