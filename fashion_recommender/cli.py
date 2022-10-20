import click
from fashion_recommender.sink.clients import FieldConfig, StructureConfig, VectorField

from fashion_recommender.sink.clients.milvus import MilvusClient, MilvusConfig

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
    click.echo(f"Setup {host}:{port}")
    config = MilvusConfig(host=host, port=port)
    client = MilvusClient(config=config)
    product_id_field = FieldConfig(name="product_id", primary_key=True)
    vector_field = VectorField(name="doc2vec_128", primary_key=False, vec_size=128)
    structure = StructureConfig(name="products", description="Products", fields=[product_id_field, vector_field])
    client.create_vector_structure(config=structure)


if __name__ == "__main__":
    cli()