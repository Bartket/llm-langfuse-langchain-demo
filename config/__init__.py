import os
from dataclasses import dataclass
from typing import Optional

@dataclass
class LangfuseConfig:
    """Configuration for Langfuse observability"""
    secret_key: str
    public_key: str
    host: str = "http://localhost:3000"

@dataclass
class ModelConfig:
    """Configuration for the Qwen model"""
    model_name: str = "Qwen/Qwen2.5-3B"
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True
    repetition_penalty: float = 1.1
    torch_dtype: str = "floatcl16"
    device_map: str = "auto"
    enable_flash_attention: bool = False

@dataclass
class AppConfig:
    """Main application configuration"""
    langfuse: LangfuseConfig
    model: ModelConfig

def load_config() -> AppConfig:
    """
    Load configuration from environment variables or defaults.
    Recommended to set LANGFUSE_SECRET_KEY and LANGFUSE_PUBLIC_KEY as environment variables.
    """
    # Load Langfuse config
    langfuse_config = LangfuseConfig(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_HOST", "http://localhost:3000")
    )

    # Load model config
    model_config = ModelConfig(
        model_name=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-3B"),
        max_length=int(os.getenv("MODEL_MAX_LENGTH", 512)),
        temperature=float(os.getenv("MODEL_TEMPERATURE", 0.7)),
        do_sample=os.getenv("MODEL_DO_SAMPLE", "true").lower() in ["true", "1", "yes"],
        repetition_penalty=float(os.getenv("MODEL_REPETITION_PENALTY", 1.1)),
        torch_dtype=os.getenv("MODEL_TORCH_DTYPE", "float16"),
        device_map=os.getenv("MODEL_DEVICE_MAP", "auto"),
        enable_flash_attention=os.getenv("MODEL_ENABLE_FLASH_ATTENTION", "false").lower() in ["true", "1", "yes"]
    )

    return AppConfig(langfuse=langfuse_config, model=model_config)