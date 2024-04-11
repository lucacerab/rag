from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
import boto3
import json

session = boto3.Session()

# OpenSearch configs
host = 'search-host.eu-west-1.es.amazonaws.com' # dummy host to be replaced
index_name = 'index-name'  # dummy index to be replaced
region = 'eu-west-1'
service = 'es'
credentials = session.get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

sagemaker_client = session.client('runtime.sagemaker')

# Using Titan Embeddings
boto3_bedrock = session.client(
    service_name = "bedrock-runtime",
    region_name = "us-east-1")

modelId = 'amazon.titan-embed-text-v1'

accept = "application/json"
contentType = "application/json"

# Create OpenSearch client
client = OpenSearch(
    hosts = [{'host': host, 'port': 443}],
    http_auth = auth,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    pool_maxsize = 20
)

def check_question_existence(question):

    payload = {"inputText": question}
    body = json.dumps(payload)
    
    response = boto3_bedrock.invoke_model(
        body=body, modelId=modelId, accept=accept, contentType=contentType
    )

    response_body = json.loads(response.get("body").read())
    embedding = response_body.get("embedding")

    k=1  
    query = {
        'size': k,
        'query': {
            'knn': {
              'vector_field': {
                'vector': embedding,
                'k': k
              }
            }
          }
        }
    
    response = client.search(
        body = query,
        index = index_name
    )

    similarity_score = float(response['hits']['hits'][0]['_score'])
    answer = response['hits']['hits'][0]['_source']['answer'] + '\n\n' + '**Fine-tuned Model**'
    image_paths = response['hits']['hits'][0]['_source']['image_paths']

    if similarity_score>0.009:
        return answer, similarity_score, image_paths
    return "Answer does not exist in the current knowledge base, triggering RAG.", similarity_score, ""
