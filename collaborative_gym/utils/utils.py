import logging
import os
import sys

import toml


def serialize_llm_response(obj):
    """
    Convert OpenAI API response objects to serializable dictionaries.
    Works recursively to handle nested objects.
    """
    if hasattr(obj, "__dict__"):
        # For custom classes, convert to dict
        return {
            k: serialize_llm_response(v)
            for k, v in obj.__dict__.items()
            if not k.startswith("_")
        }
    elif isinstance(obj, dict):
        # Recursively serialize dictionary values
        return {k: serialize_llm_response(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Recursively serialize list items
        return [serialize_llm_response(item) for item in obj]
    elif isinstance(obj, (str, int, float, bool, type(None))):
        # Basic types can be returned as is
        return obj
    else:
        # For any other objects, convert to string representation
        return str(obj)


def load_api_key(toml_file_path):
    try:
        with open(toml_file_path, "r") as file:
            data = toml.load(file)
    except FileNotFoundError:
        print(f"File not found: {toml_file_path}", file=sys.stderr)
        return
    except toml.TomlDecodeError:
        print(f"Error decoding TOML file: {toml_file_path}", file=sys.stderr)
        return
    # Set environment variables
    for key, value in data.items():
        os.environ[key] = str(value)


def prepare_lm_kwargs(model_name):
    """Prepare kwargs for initializing a language model using knowledge_storm.lm.LitellmModel."""
    if "azure" in model_name:
        api_key = os.environ["AZURE_API_KEY"]
    elif "gpt" in model_name:
        api_key = os.environ["OPENAI_API_KEY"]
    elif "claude" in model_name:
        api_key = os.environ["ANTHROPIC_API_KEY"]
    elif "together_ai" in model_name:
        api_key = os.environ["TOGETHER_API_KEY"]
    elif "deepseek" in model_name:
        api_key = os.environ["DEEPSEEK_API_KEY"]
    elif "gemini" in model_name:
        api_key = os.environ["GEMINI_API_KEY"]
    else:
        logging.warning(
            'API key not found in environment variables. Using os.environ["API_KEY"].'
        )
        api_key = os.environ["API_KEY"]
    lm_kwargs = {"model": model_name, "api_key": api_key}
    if "azure" in model_name:
        lm_kwargs["api_base"] = os.environ["AZURE_ENDPOINT"]
        lm_kwargs["api_version"] = os.environ["AZURE_API_VERSION"]

    return lm_kwargs
