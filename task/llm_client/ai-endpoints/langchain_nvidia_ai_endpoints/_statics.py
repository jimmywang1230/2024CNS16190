import os
from typing import Any, Dict, Optional

from langchain_core.pydantic_v1 import BaseModel, root_validator, Field


class Metadata(BaseModel):
    infer_args: dict = Field({})
    client_args: dict = Field({})
        

class Model(BaseModel):
    id: str
    model_type: Optional[str] = None
    metadata: Optional[Metadata] = None
    # path: str

    @root_validator(pre=True)
    def build_extra(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        all_required_field_names = {field.alias for field in cls.__fields__.values() if field.alias != 'metadata'}
        client_args = ["client", "base_url", "infer_path", "api_key_var", "api_type", "mode", "alternative"]
        client_kw: Dict[str, Any] = {}
        infer_kw: Dict[str, Any] = {}
        for field_name in list(values):
            if field_name in client_args:
                client_kw[field_name] = values.pop(field_name)
            elif field_name not in all_required_field_names:
                infer_kw[field_name] = values.pop(field_name)
        values['metadata'] = Metadata(client_args=client_kw, infer_args=infer_kw)
        return values


NVCF_PG_SPECS = {
    "playground_smaug_72b": {"model_type": "chat", "api_type": "aifm"},
    "playground_kosmos_2": {"model_type": "vlm", "api_type": "aifm"},
    "playground_llama2_70b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-llama2-70b",
    },
    "playground_nvolveqa_40k": {"model_type": "embedding", "api_type": "aifm"},
    "playground_nemotron_qa_8b": {"model_type": "qa", "api_type": "aifm"},
    "playground_gemma_7b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-gemma-7b",
    },
    "playground_mistral_7b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-mistral-7b-instruct-v2",
    },
    "playground_mamba_chat": {"model_type": "chat", "api_type": "aifm"},
    "playground_phi2": {"model_type": "chat", "api_type": "aifm"},
    "playground_sdxl": {"model_type": "genai", "api_type": "aifm"},
    "playground_nv_llama2_rlhf_70b": {"model_type": "chat", "api_type": "aifm"},
    "playground_neva_22b": {
        "model_type": "vlm",
        "api_type": "aifm",
        "alternative": "ai-neva-22b",
    },
    "playground_yi_34b": {"model_type": "chat", "api_type": "aifm"},
    "playground_nemotron_steerlm_8b": {"model_type": "chat", "api_type": "aifm"},
    "playground_cuopt": {"model_type": "cuopt", "api_type": "aifm"},
    "playground_llama_guard": {"model_type": "classifier", "api_type": "aifm"},
    "playground_starcoder2_15b": {"model_type": "completion", "api_type": "aifm"},
    "playground_deplot": {
        "model_type": "vlm",
        "api_type": "aifm",
        "alternative": "ai-google-deplot",
    },
    "playground_llama2_code_70b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-codellama-70b",
    },
    "playground_gemma_2b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-gemma-2b",
    },
    "playground_seamless": {"model_type": "translation", "api_type": "aifm"},
    "playground_mixtral_8x7b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-mixtral-8x7b-instruct",
    },
    "playground_fuyu_8b": {
        "model_type": "vlm",
        "api_type": "aifm",
        "alternative": "ai-fuyu-8b",
    },
    "playground_llama2_code_34b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-codellama-70b",
    },
    "playground_llama2_code_13b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-codellama-70b",
    },
    "playground_steerlm_llama_70b": {"model_type": "chat", "api_type": "aifm"},
    "playground_clip": {"model_type": "similarity", "api_type": "aifm"},
    "playground_llama2_13b": {
        "model_type": "chat",
        "api_type": "aifm",
        "alternative": "ai-llama2-70b",
    },
}

NVCF_AI_SPECS = {
    "ai-codellama-70b": {"model_type": "chat", "model_name": "meta/codellama-70b"},
    "ai-embed-qa-4": {"model_type": "embedding", "model_name": "NV-Embed-QA"},
    "ai-fuyu-8b": {"model_type": "vlm"},
    "ai-gemma-7b": {"model_type": "chat", "model_name": "google/gemma-7b"},
    "ai-google-deplot": {"model_type": "vlm"},
    "ai-llama2-70b": {"model_type": "chat", "model_name": "meta/llama2-70b"},
    "ai-microsoft-kosmos-2": {"model_type": "vlm"},
    "ai-mistral-7b-instruct-v2": {
        "model_type": "chat",
        "model_name": "mistralai/mistral-7b-instruct-v0.2",
    },
    "ai-mixtral-8x7b-instruct": {
        "model_type": "chat",
        "model_name": "mistralai/mixtral-8x7b-instruct-v0.1",
    },
    "ai-neva-22b": {"model_type": "vlm"},
    "ai-rerank-qa-mistral-4b": {
        "model_type": "ranking",
        "model_name": "nv-rerank-qa-mistral-4b:1",  # nvidia/rerank-qa-mistral-4b
    },
    'ai-sdxl-turbo': {'model_type': 'genai'},
    'ai-stable-diffusion-xl-base': {'model_type': 'iamge_out'},
    "ai-codegemma-7b": {"model_type": "chat", "model_name": "google/codegemma-7b"},
    "ai-recurrentgemma-2b": {
        "model_type": "chat",
        "model_name": "google/recurrentgemma-2b",
    },
    "ai-gemma-2b": {"model_type": "chat", "model_name": "google/gemma-2b"},
    "ai-mistral-large": {
        "model_type": "chat",
        "model_name": "mistralai/mistral-large",
    },
    "ai-mixtral-8x22b": {
        "model_type": "completion",
        "model_name": "mistralai/mixtral-8x22b-v0.1",
    },
    "ai-mixtral-8x22b-instruct": {
        "model_type": "chat",
        "model_name": "mistralai/mixtral-8x22b-instruct-v0.1",
    },
    "ai-llama3-8b": {"model_type": "chat", "model_name": "meta/llama3-8b-instruct"},
    "ai-llama3-70b": {
        "model_type": "chat",
        "model_name": "meta/llama3-70b-instruct",
    },
    "ai-phi-3-mini": {
        "model_type": "chat",
        "model_name": "microsoft/phi-3-mini-128k-instruct",
    },
    "ai-arctic": {"model_type": "chat", "model_name": "snowflake/arctic"},
    "ai-dbrx-instruct": {
        "model_type": "chat",
        "model_name": "databricks/dbrx-instruct",
    },
}

CATALOG_SPECS = {
    'aisingapore/sea-lion-7b-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'databricks/dbrx-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'google/codegemma-7b': {'model_type': 'chat', 'max_tokens': 2048},
    'google/gemma-2b': {'model_type': 'chat', 'max_tokens': 2048},
    'google/gemma-7b': {'model_type': 'chat', 'max_tokens': 2048},
    'google/recurrentgemma-2b': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/codellama-70b': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/llama2-70b': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/llama3-70b-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'meta/llama3-8b-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'microsoft/phi-3-mini-128k-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'microsoft/phi-3-mini-4k-instruct': {'model_type': 'chat', 'max_tokens': 2048},
    'mistralai/mistral-7b-instruct-v0.2': {'model_type': 'chat', 'max_tokens': 2048},
    'mistralai/mistral-large': {'model_type': 'chat', 'max_tokens': 2048}, 
    'mistralai/mixtral-8x22b-instruct-v0.1': {'model_type': 'chat', 'max_tokens': 2048},
    ## TODO: Very temporary behavior. Expect it to change
    'mistralai/mixtral-8x22b-v0.1': {'model_type': 'completion', 'max_tokens': 2048, 'infer_path': '{base_url}/chat/completions'},
    'mistralai/mixtral-8x7b-instruct-v0.1': {'model_type': 'chat', 'max_tokens': 2048},
    'seallms/seallm-7b-v2.5': {'model_type': 'chat', 'max_tokens': 2048},
    'snowflake/arctic': {'model_type': 'chat', 'max_tokens': 2048},
    'nvidia/embed-qa-4': {
        'model_type': 'embedding', 
        'base_url': 'https://ai.api.nvidia.com/v1', 
        'infer_path': '{base_url}/retrieval/nvidia/embeddings', 
        'model_name': 'NV-Embed-QA'
    },
    "nvidia/rerank-qa-mistral-4b": {
        "model_type": "ranking",
        "model_name": "nv-rerank-qa-mistral-4b:1",
        'base_url': 'https://ai.api.nvidia.com/v1', 
        'infer_path': '{base_url}/retrieval/nvidia/reranking', 
    },
    'snowflake/arctic-embed-l': {
        'model_type': 'embedding', 
        'base_url': 'https://ai.api.nvidia.com/v1', 
        'infer_path': '{base_url}/retrieval/{model_name}/embeddings'
    },
    'adept/fuyu-8b': {'model_type': 'vlm'},
    'google/deplot': {'model_type': 'vlm'},
    'microsoft/kosmos-2': {'model_type': 'vlm'},
    'nvidia/neva-22b': {'model_type': 'vlm'},
    "stabilityai/stable-diffusion-xl": {'model_type': 'genai'},
    "stabilityai/sdxl-turbo": {'model_type': 'genai'},
    "stabilityai/stable-video-diffusion": {'model_type': 'genai'},
}

OPENAI_SPECS = {
    "babbage-002": {"model_type": "completion"},
    "dall-e-2": {"model_type": "genai"},
    "dall-e-3": {"model_type": "genai"},
    "davinci-002": {"model_type": "completion"},
    "gpt-3.5-turbo-0125": {"model_type": "chat"},
    "gpt-3.5-turbo-0301": {"model_type": "chat"},
    "gpt-3.5-turbo-0613": {"model_type": "chat"},
    "gpt-3.5-turbo-1106": {"model_type": "chat"},
    "gpt-3.5-turbo-16k-0613": {"model_type": "chat"},
    "gpt-3.5-turbo-16k": {"model_type": "chat"},
    "gpt-3.5-turbo-instruct-0914": {"model_type": "completion"},
    "gpt-3.5-turbo-instruct": {"model_type": "completion"},
    "gpt-3.5-turbo": {"model_type": "chat"},
    "gpt-4": {"model_type": "chat"},
    "gpt-4-0125-preview": {"model_type": "chat"},
    "gpt-4-0613": {"model_type": "chat"},
    "gpt-4-1106-preview": {"model_type": "chat"},
    "gpt-4-1106-vision-preview": {"model_type": "chat"},
    "gpt-4-turbo": {"model_type": "chat"},
    "gpt-4-turbo-2024-04-09": {"model_type": "chat"},
    "gpt-4-turbo-preview": {"model_type": "chat"},
    "gpt-4-vision-preview": {"model_type": "chat"},
    "text-embedding-3-large": {"model_type": "embedding"},
    "text-embedding-3-small": {"model_type": "embedding"},
    "text-embedding-ada-002": {"model_type": "embedding"},
    "tts-1-1106": {"model_type": "tts"},
    "tts-1-hd-1106": {"model_type": "tts"},
    "tts-1-hd": {"model_type": "tts"},
    "tts-1": {"model_type": "tts"},
    "whisper-1": {"model_type": "asr"},
}

CLIENT_MAP = {
    "asr": "RunnableNVIDIA",
    "chat": "ChatNVIDIA",
    "classifier": "RunnableNVIDIA",
    "completion": "NVIDIA",
    "cuopt": "RunnableNVIDIA",
    "embedding": "NVIDIAEmbeddings",
    "vlm": "ChatNVIDIA",
    "genai": "RunnableNVIDIA",
    "qa": "ChatNVIDIA",
    "similarity": "RunnableNVIDIA",
    "translation": "RunnableNVIDIA",
    "tts": "RunnableNVIDIA",
    "ranking": "NVIDIARerank",
}

for mname, mspec in CATALOG_SPECS.items():
    base_url = ""
    infer_path = ""
    if mspec.get('model_type') == 'vlm':
        base_url = 'https://ai.api.nvidia.com/v1'
        infer_path = '{base_url}/vlm/{model_name}'
    elif mspec.get('model_type') == 'genai':
        base_url = 'https://ai.api.nvidia.com/v1'
        infer_path = '{base_url}/genai/{model_name}'
    elif mspec.get('model_type') == 'embeddings':
        base_url = 'https://ai.api.nvidia.com/v1'
        infer_path = '{base_url}/retrieval/{model_name}'
    if base_url and 'base_url' not in mspec:
        mspec['base_url'] = base_url
    if infer_path and 'infer_path' not in mspec:
        mspec['infer_path'] = infer_path

tooled_models = ["gpt-4"]

SPEC_LIST = [CATALOG_SPECS, OPENAI_SPECS, NVCF_PG_SPECS, NVCF_AI_SPECS]
MODE_LIST = ["nvidia", "openai", "nvcf", "nvcf"]

for spec, mode in zip(SPEC_LIST, MODE_LIST):
    for mname, mspec in spec.items():
        mspec["mode"] = mspec.get("mode") or mode
        ## Default max_tokens for models
        if mspec.get('model_type') in ('chat', 'vlm'):
            if 'max_tokens' not in mspec:
                mspec['max_tokens'] = 1024
        ## Default Client enforcement
        mspec['client'] = mspec.get("client") or [CLIENT_MAP.get(mspec.get("model_type"))]
        if not isinstance(mspec['client'], list):
            mspec['client'] = [mspec['client']]
        if mname in tooled_models:
            mspec['client'] += [f"Tooled{client}" for client in mspec['client']]

OPEN_SPECS = {**CATALOG_SPECS, **OPENAI_SPECS}

for mname, mspec in OPEN_SPECS.items():
    if not mspec.get('infer_path'):
        model_type = mspec.get('model_type')
        if model_type == 'chat':
            mspec['infer_path'] = '{base_url}/chat/completions'
        if model_type == 'completion':
            mspec['infer_path'] = '{base_url}/completions'
        if model_type == 'embedding':
            mspec['infer_path'] = '{base_url}/embeddings'
        if model_type == 'genai':
            mspec['infer_path'] = '{base_url}/images/generations'

NVCF_SPECS = {**NVCF_PG_SPECS, **NVCF_AI_SPECS}

MODEL_SPECS = {**OPEN_SPECS, **NVCF_SPECS}