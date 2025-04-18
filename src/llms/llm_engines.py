import vertexai
import os

from dotenv import load_dotenv
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel
from vertexai.generative_models import GenerationConfig
from anthropic import AnthropicVertex

load_dotenv(override=True)

PROJECT = os.getenv("GCP_PROJECT")
REGION = os.getenv("GCP_REGION")
GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")
scope = "https://www.googleapis.com/auth/cloud-platform" 

aiplatform.init(
  project=PROJECT,
  location=REGION,
  credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
)
vertexai.init(project=PROJECT, location=REGION, credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS))

def call_model(model_name: str, prompt: str, temperature: float = 0.2, max_output_tokens: int = 2048):
    """
    Get a Vertex AI model instance.
    """
    if model_name == "gemini-2.0-flash":
        model = GenerativeModel("gemini-2.0-flash")
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return model.generate_content(contents=[prompt], generation_config=generation_config).candidates[0].text
    elif model_name == "gemini-1.5-pro-002":
        model = GenerativeModel("gemini-1.5-pro-002")
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return model.generate_content(contents=[prompt], generation_config=generation_config).candidates[0].text
    elif model_name == "gemini-1.5-flash-002":
        model = GenerativeModel("gemini-1.5-flash-002")
        generation_config = GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return model.generate_content(contents=[prompt], generation_config=generation_config).candidates[0].text
    elif model_name == "claude-3-7-sonnet":
        client = AnthropicVertex(region="us-east5", project_id=PROJECT, credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS, scopes=[scope]))
        message = client.messages.create(
            max_tokens=max_output_tokens,
            temperature=temperature,
            messages=[
            {
                "role": "user",
                "content": prompt
            }
            ],
            model="claude-3-7-sonnet@20250219"
        )
        return message.content[0].text
    elif model_name == "claude-3-5-sonnet":
        client = AnthropicVertex(region="us-east5", project_id=PROJECT, credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS, scopes=[scope]))
        message = client.messages.create(
            max_tokens=max_output_tokens,
            temperature=temperature,
            messages=[
            {
                "role": "user",
                "content": prompt
            }
            ],
            model="claude-3-5-sonnet-v2@20241022"
        )
        return message.content[0].text
    else:
        raise ValueError(f"Model {model_name} not supported.")

