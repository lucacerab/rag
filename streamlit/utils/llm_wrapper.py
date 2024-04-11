from langchain.llms.sagemaker_endpoint import LLMContentHandler
from langchain import SagemakerEndpoint
import json

llm_endpoint_name = "hf-llm-mistral-7b-instruct"

class ContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: dict) -> bytes:
        input_str = json.dumps(
            {
                "inputs" : "<s>[INST]" + prompt + "[/INST]",
                "parameters" : {**model_kwargs}
            }
        )
        return input_str.encode('utf-8')
    
    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]['generated_text']
    
content_handler = ContentHandler()

sagemaker_llm=SagemakerEndpoint(
     endpoint_name=llm_endpoint_name,
     region_name="eu-west-1", 
     model_kwargs={"max_new_tokens": 700, "top_p": 0.9, "temperature": 0.3},
     content_handler=content_handler
 )