import os
from dotenv import load_dotenv

import vertexai
from google.oauth2 import service_account
from google.cloud import aiplatform
from vertexai.generative_models import GenerativeModel, GenerationConfig

from anthropic import AnthropicVertex
from openai import OpenAI 

load_dotenv(override=True)

# Existing environment variables for GCP
PROJECT = os.getenv("GCP_PROJECT")
REGION = os.getenv("GCP_REGION")
GCP_CREDENTIALS = os.getenv("GCP_CREDENTIALS")

# New environment variable for OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize GCP clients
aiplatform.init(
    project=PROJECT,
    location=REGION,
    credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
)
vertexai.init(
    project=PROJECT,
    location=REGION,
    credentials=service_account.Credentials.from_service_account_file(GCP_CREDENTIALS)
)

def call_model(model_name: str, prompt: str, temperature: float = 0.2, max_output_tokens: int = 2048):
    """
    Dispatch to the requested model.
    """
    # --- Vertex AI Models ---
    if model_name == "gemini-2.0-flash":
        model = GenerativeModel("gemini-2.0-flash")
        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        return model.generate_content([prompt], generation_config=cfg).candidates[0].text

    elif model_name == "gemini-1.5-pro-002":
        model = GenerativeModel("gemini-1.5-pro-002")
        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        return model.generate_content([prompt], generation_config=cfg).candidates[0].text

    elif model_name == "gemini-1.5-flash-002":
        model = GenerativeModel("gemini-1.5-flash-002")
        cfg = GenerationConfig(temperature=temperature, max_output_tokens=max_output_tokens)
        return model.generate_content([prompt], generation_config=cfg).candidates[0].text

    # --- Anthropic Claude Models ---
    elif model_name in ("claude-3-7-sonnet", "claude-3-5-sonnet"):
        # Choose the correct Anthropic model tag
        anthro_model = {
            "claude-3-7-sonnet": "claude-3-7-sonnet@20250219",
            "claude-3-5-sonnet": "claude-3-5-sonnet-v2@20241022"
        }[model_name]

        client = AnthropicVertex(
            region="us-east5",
            project_id=PROJECT,
            credentials=service_account.Credentials.from_service_account_file(
                GCP_CREDENTIALS,
                scopes=["https://www.googleapis.com/auth/cloud-platform"]
            )
        )
        resp = client.messages.create(
            model=anthro_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_output_tokens,
            temperature=temperature,
        )
        return resp.content[0].text

    # --- OpenAI GPT-4o Models ---
    elif model_name in ("gpt-4o", "gpt-4o-mini"):
        # Initialize OpenAI client
        client = OpenAI(api_key=OPENAI_API_KEY)  # <-- uses env var per best practices :contentReference[oaicite:2]{index=2}

        # Request a chat completion
        completion = client.chat.completions.create(
            model=model_name,                   # "gpt-4o" or "gpt-4o-mini"
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,            # same parameter name
            max_tokens=max_output_tokens        # analogous to max_output_tokens :contentReference[oaicite:3]{index=3}
        )
        return completion.choices[0].message.content

    else:
        BASE_URL = os.getenv("BASE_URL")
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)

        # Request a chat completion
        completion = client.chat.completions.create(
            model=model_name,                   # "gpt-4o" or "gpt-4o-mini"
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,            # same parameter name
            max_tokens=max_output_tokens        # analogous to max_output_tokens :contentReference[oaicite:3]{index=3}
        )
        return completion.choices[0].message.content
