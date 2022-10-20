from typing import Dict, Optional
from pymilvus import (
    Collection,
    connections,
    CollectionSchema,
    FieldSchema,
    DataType
)

from dataclasses import dataclass
from fashion_recommender.sink.clients import (
    FieldConfig,
    StringField,
    StructureConfig,
    VectorField
)

@dataclass
class MilvusConfig:
    host: str
    port: int
    user: Optional[str] = None
    password: Optional[str] = None
    
    @property
    def params(self) -> dict:
        value =  {
            "host": self.host,
            "port": self.port,
        }
        if self.user and self.password:
            value["user"] = self.user
            value["password"] = self.password
        
        return value


class MilvusClient:
    _connection_count: int = 0
    _MAPPED_TYPES: Dict[str, DataType] = {
        FieldConfig: DataType.INT64,
        VectorField: DataType.FLOAT_VECTOR,
        StringField: DataType.VARCHAR
    }

    def __init__(self, config: MilvusConfig) -> None:
        # connections is a singleton instance of a Connections class
        # from the milvus library
        self._alias = f"default_{self._connection_count}"
        self._connection = connections.connect(
            alias=self._alias,
            **config.params
        )
        self._connection_count += 1
        self._structures = {}
    

    def create_vector_structure(self, config: StructureConfig) -> None:
        def _parse_conf(conf: FieldConfig) -> FieldSchema:
            extra_params = {}
            if type(conf) is VectorField:
                extra_params["dim"] = conf.vec_size
            elif type(conf) is StringField:
                extra_params["max_length"] = conf.max_length
            else:
                extra_params["is_primary"] = conf.primary_key

            return FieldSchema(
                name=conf.name,
                dtype=self._MAPPED_TYPES[type(conf)],
                **extra_params
            )

        fields = list(map(_parse_conf, config.fields))
        schema = CollectionSchema(
            fields=fields,
            description=config.description
        )
        collection = Collection(
            name=config.name,
            schema=schema,
            using=self._alias
        )
        if config.partitions:
            for partition in config.partitions:
                collection.create_partition(partition_name=partition)
        
        if config.index_by:
            collection.create_index(field_name=config.index_by)

        self._structures[config.name] = collection