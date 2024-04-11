from opensearchpy import RequestsHttpConnection, AWSV4SignerAuth
from langchain.vectorstores import OpenSearchVectorSearch
from langchain.vectorstores import base
from langchain.callbacks.manager import CallbackManagerForRetrieverRun
from typing import List
from langchain.docstore.document import Document
from utils.embeddings_wrapper import sagemaker_embeddings
import boto3

session = boto3.Session()

# OpenSearch configs
host = 'https://search-host.eu-west-1.es.amazonaws.com' # dummy host to be replaced
domain_index = 'index-name' # dummy index to be replaced
region = 'eu-west-1'
service = 'es'

credentials = session.get_credentials()
auth = AWSV4SignerAuth(credentials, region, service)

# E5 - Embeddings SageMaker endpoint
embeddings = sagemaker_embeddings

# Create OpenSearch client
docsearch = OpenSearchVectorSearch(
    index_name=domain_index,
    opensearch_url=host,
    embedding_function=embeddings,
    http_auth=auth,
    timeout=300,
    use_ssl = True,
    verify_certs = True,
    connection_class = RequestsHttpConnection,
    is_aoss=False
)

# Need to create a class to inject score from opensearch in documents' metadata cause ConversationRetrievalChain doesn't have score support by default
class VectorStoreRetrieverWithScore(base.VectorStoreRetriever):
    # See https://github.com/langchain-ai/langchain/blob/61dd92f8215daef3d9cf1734b0d1f8c70c1571c3/libs/langchain/langchain/vectorstores/base.py#L500
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        docs_and_similarities = (
            self.vectorstore.similarity_search_with_score(
                query, **self.search_kwargs
            )
        )

        # Make the score part of the document metadata
        for doc, similarity in docs_and_similarities:
            doc.metadata["score"] = similarity

        docs = [doc for doc, _ in docs_and_similarities]
        return docs