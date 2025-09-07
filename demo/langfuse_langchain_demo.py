import sys
sys.path.append('../')
import os
import time
import torch
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Optional

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langfuse.langchain import CallbackHandler
from langfuse import Langfuse

# Load .env and set as system environment variables
load_dotenv(override=True)

# Enable Langfuse debug logs
os.environ["LANGFUSE_DEBUG"] = "true"

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
    torch_dtype: str = "float16"
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

# Load configuration
config = load_config()

# Initialize Langfuse
langfuse = Langfuse(
    public_key=config.langfuse.public_key,
    secret_key=config.langfuse.secret_key,
    host=config.langfuse.host
)

# Create Langfuse callback handler
langfuse_handler = CallbackHandler()

print(f"[Langfuse] Initialized with host={config.langfuse.host}")
print(f"[Langfuse] Public key: {config.langfuse.public_key[:10]}...")
print(f"[Langfuse] Secret key configured: {'Yes' if config.langfuse.secret_key else 'No'}")

class LangChainQwenChatbot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load the Qwen model with same configuration
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name, trust_remote_code=True
        )
        
        # Prepare model kwargs
        torch_dtype = getattr(torch, config.model.torch_dtype, None)
        model_kwargs = {"device_map": config.model.device_map, "trust_remote_code": True}
        if torch_dtype: 
            model_kwargs["torch_dtype"] = torch_dtype
        if config.model.enable_flash_attention:
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name, **model_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Create a transformers pipeline (no device argument when using accelerate/device_map)
        self.pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_length=config.model.max_length,
            temperature=config.model.temperature,
            do_sample=config.model.do_sample,
            repetition_penalty=config.model.repetition_penalty,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            return_full_text=False  # Only return generated text, not the prompt
        )
        
        # Wrap the pipeline in LangChain's HuggingFacePipeline
        self.llm = HuggingFacePipeline(
            pipeline=self.pipeline,
            callbacks=[langfuse_handler]
        )
        
        # Create a simple prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["system_message", "user_message"],
            template="System: {system_message}\nUser: {user_message}\nAssistant:"
        )
        
        # Create a chain
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template,
            callbacks=[langfuse_handler]
        )

    def simple_chat(self, user_message, system_message="You are a helpful AI assistant."):
        """Simple chat using the LLM directly"""
        prompt = f"System: {system_message}\nUser: {user_message}\nAssistant:"
        
        response = self.llm.invoke(prompt, config={"callbacks": [langfuse_handler]})
        return response.strip()

    def chain_chat(self, user_message, system_message="You are a helpful AI assistant."):
        """Chat using LangChain chain"""
        response = self.chain.run(
            system_message=system_message,
            user_message=user_message,
            callbacks=[langfuse_handler]
        )
        return response.strip()

    def advanced_chain(self, topic):
        """More complex chain with multiple steps"""
        prompt = PromptTemplate(
            input_variables=["topic"],
            template="System: You are a helpful AI assistant.\nUser: Write a brief explanation about {topic} in exactly 3 sentences.\nAssistant:"
        )
        
        chain = (
            RunnablePassthrough() 
            | prompt 
            | self.llm
        )
        
        response = chain.invoke(
            {"topic": topic}, 
            config={"callbacks": [langfuse_handler]}
        )
        
        return response.strip()

def run_langchain_demo():
    """Run the LangChain + Langfuse demo with Qwen model"""
    chatbot = LangChainQwenChatbot()
    print("LangChain + Langfuse Qwen Chatbot initialized successfully!")

    # Example 1: Simple chat
    print("\n=== SIMPLE CHAT DEMO ===")
    examples = [
        "What is the capital of France?",
        "Explain machine learning in simple terms",
        "What are the benefits of renewable energy?"
    ]
    
    for question in examples:
        print(f"\nQuestion: {question}")
        response = chatbot.simple_chat(question)
        print(f"Response: {response}")

    # Example 2: Chain-based chat
    print("\n=== CHAIN CHAT DEMO ===")
    chain_questions = [
        "How does blockchain technology work?",
        "What is quantum computing?"
    ]
    
    for question in chain_questions:
        print(f"\nChain Question: {question}")
        response = chatbot.chain_chat(question)
        print(f"Chain Response: {response}")

    # Example 3: Advanced chain
    print("\n=== ADVANCED CHAIN DEMO ===")
    topics = ["artificial intelligence", "climate change", "space exploration"]
    
    for topic in topics:
        print(f"\nTopic: {topic}")
        response = chatbot.advanced_chain(topic)
        print(f"Advanced Response: {response}")

def batch_processing_demo():
    """Demo batch processing with LangChain and Qwen"""
    print("\n=== BATCH PROCESSING DEMO ===")
    chatbot = LangChainQwenChatbot()
    
    batch_questions = [
        "What is Python programming?",
        "How do neural networks work?",
        "Explain the concept of APIs",
        "What is cloud computing?"
    ]
    
    results = []
    start_time = time.time()
    
    for i, question in enumerate(batch_questions, 1):
        print(f"Processing batch {i}/{len(batch_questions)}: {question}")
        
        response = chatbot.simple_chat(question)
        results.append({"id": i, "question": question, "response": response})
    
    total_time = time.time() - start_time
    print(f"\nBatch processing completed in {total_time:.2f} seconds")
    return results

def performance_test():
    """Performance test with detailed Langfuse tracking"""
    print("\n=== PERFORMANCE TEST ===")
    chatbot = LangChainQwenChatbot()
    
    # Warmup
    chatbot.simple_chat("Hello")
    
    # Actual test
    test_prompt = "Explain the importance of sustainable development in modern society"
    
    start_time = time.time()
    response = chatbot.simple_chat(test_prompt)
    end_time = time.time()
    
    duration = end_time - start_time
    
    print(f"Test completed in {duration:.2f} seconds")
    print(f"Response length: {len(response)} characters")
    print(f"Response: {response}")
    
    # Add GPU memory info if available
    if torch.cuda.is_available():
        print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"GPU Memory reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")
    
    return {
        "duration": duration,
        "response": response,
        "response_length": len(response)
    }

if __name__ == "__main__":
    try:
        # Run all demos
        run_langchain_demo()
        batch_processing_demo()
        performance_test()
        
        print("\nFlushing Langfuse observations...")
        langfuse.flush()
        print("✅ Successfully flushed to Langfuse!")
        
    except Exception as e:
        print(f"Error running demo: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Ensure cleanup
        try:
            langfuse_handler.flush()
        except:
            pass