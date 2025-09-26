from schemas import GetInformationInput
from langchain_core.tools import tool
import boto3
from typing import Tuple, List, Dict
from dotenv import load_dotenv
import json
import os

load_dotenv()

S3_VECTOR_BUCKET_NAME=os.getenv("S3_VECTOR_BUCKET_NAME")
UMM_S3_VECTOR_INDEX=os.getenv("UMM_S3_VECTOR_INDEX")

def embed_query(query: str) -> List[Tuple[List[float], str]]:

    bedrock_client = boto3.client("bedrock-runtime")

    response = response = bedrock_client.invoke_model(
        modelId="amazon.titan-embed-text-v2:0",
        body=json.dumps({"inputText": query}),
    )

    response_body = json.loads(response["body"].read())

    return response_body["embedding"]

@tool(args_schema=GetInformationInput)
def get_information_tool(query: str, modality: str, program_type: str) -> str:
    """Use this tool to find general information about the Universidad Metropolitana de Monterrey (UMM), 
    its study programs, scholarships, financing, admissions information, or any related topic..
    """

    s3vectors_client = boto3.client("s3vectors")

    query_vector = embed_query(query)
    query_filter = []

    if modality:
        modality_filter = {"program_modalities": {"$in": modality}}
        query_filter.append(modality_filter)
    if program_type:
        program_type_filter = {"program_type": program_type}
        query_filter.append(program_type_filter)


    response: Dict = s3vectors_client.query_vectors(
        vectorBucketName=S3_VECTOR_BUCKET_NAME,
        indexName=UMM_S3_VECTOR_INDEX,
        topK=5,
        queryVector={
            'float32': query_vector
        },
        returnMetadata=True,
        returnDistance=True
    )

    vectors = response.get("vectors", [])
    if not vectors:
        return "No se encontró información relevante o ocurrió un error. Intentar de nuevo"
    
    parsed_results = "\n\n".join(vector["metadata"]["source_text"] for vector in vectors)

    return f"Estos son los resultados mas relevantes con respeto a tu consulta:\n\n{parsed_results}"
