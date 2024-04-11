import os
import json
from typing import Dict, List
from langchain.embeddings.sagemaker_endpoint import EmbeddingsContentHandler
from langchain.embeddings import SagemakerEndpointEmbeddings

embedding_endpoint_name = 'huggingface-textembedding-e5-large-v2-1688995087'
region = 'eu-west-1'

class ContentHandler(EmbeddingsContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, inputs: list[str], model_kwargs: Dict) -> bytes:
        input_str = json.dumps({"inputs": inputs, **model_kwargs})
        return input_str.encode("utf-8")

    def transform_output(self, output: bytes) -> List[List[float]]:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json["vectors"]
    
content_handler = ContentHandler()
    
sagemaker_embeddings = SagemakerEndpointEmbeddings(
    endpoint_name=embedding_endpoint_name,
    region_name=region,
    content_handler=content_handler,
)