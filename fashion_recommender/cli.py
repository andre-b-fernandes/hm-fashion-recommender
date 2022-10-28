import click
from fashion_recommender.models.deserializers.disk import DiskDeserializer
from fashion_recommender.models.doc2vec import Doc2Vec
from fashion_recommender.pipelines.predictions import PredictionPipeline, PredictionSinkPipeline
from fashion_recommender.sink.clients import FieldConfig, StructureConfig, VectorField

from fashion_recommender.sink.clients.milvus import MilvusClient, MilvusConfig
from fashion_recommender.sink.vector import VectorSink

@click.group()
def cli():
    pass

@cli.group()
def sinks():
    pass

@sinks.group()
def milvus():
    click.echo("Milvus")

@milvus.command()
@click.argument("host")
@click.argument("port")
def setup(host: str, port: str):
    config = MilvusConfig(host=host, port=port)
    client = MilvusClient(config=config)
    product_id_field = FieldConfig(name="product_id", primary_key=True)
    vector_field = VectorField(name="doc2vec_128", primary_key=False, vec_size=128)
    structure = StructureConfig(name="products", description="Products", fields=[product_id_field, vector_field])
    client.create_vector_structure(config=structure)

@cli.group()
def pipelines():
    pass

@pipelines.command()
@click.argument("model")
@click.argument("sink")
@click.option("--host")
@click.option("--port")
@click.option("--structure")
def predictions(model: str, sink: str, host: str, port: str, structure: str):
    config = MilvusConfig(host=host, port=port)
    client = MilvusClient(config=config)
    sink = VectorSink(client=client)
    deserializer = DiskDeserializer()
    pipeline = PredictionSinkPipeline(model_type=Doc2Vec, deserializer=deserializer, sink=sink)
    pipeline.run(to_predict=pipeline.model.extras["pid_encoder"].classes_, structure_name=structure, batch_size=128)


if __name__ == "__main__":
    cli()