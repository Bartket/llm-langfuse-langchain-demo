import sys
sys.path.append('../')
import time
import traceback
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from langfuse import Langfuse, observe, get_client
from dotenv import load_dotenv
from config import load_config

# Load .env and set as system environment variables
load_dotenv(override=True)

# Enable Langfuse debug logs
os.environ["LANGFUSE_DEBUG"] = "true"

# Load configuration
config = load_config()

print(config.langfuse.public_key, config.langfuse.secret_key, config.langfuse.host
)

# Initialize the Langfuse client once at module level
langfuse = Langfuse(
    public_key=config.langfuse.public_key,
    secret_key=config.langfuse.secret_key,
    host=config.langfuse.host
)
print(f"[Langfuse] Initialized with host={config.langfuse.host}")

# Define no-op fallback in case Langfuse fails at runtime
_langfuse_enabled = True

class QwenChatbot:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.model_name, trust_remote_code=True
        )

        # Prepare model kwargs
        torch_dtype = getattr(torch, config.model.torch_dtype, None)
        model_kwargs = {"device_map": config.model.device_map, "trust_remote_code": True}
        if torch_dtype: model_kwargs["torch_dtype"] = torch_dtype
        if getattr(config.model, "enable_flash_attention", False):
            model_kwargs["attn_implementation"] = "flash_attention_2"

        self.model = AutoModelForCausalLM.from_pretrained(
            config.model.model_name, **model_kwargs
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @observe(name="qwen_generate", as_type="generation")
    def generate_response(self, prompt, max_length=None, temperature=None, do_sample=None):
        try:
            max_length = max_length or config.model.max_length
            temperature = temperature or config.model.temperature
            do_sample = do_sample if do_sample is not None else config.model.do_sample

            inputs = self.tokenizer(prompt, return_tensors="pt", padding=True).to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=getattr(config.model, "repetition_penalty", 1.0)
                )

            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response.startswith(prompt):
                response = response[len(prompt):].strip()
            return response
        except Exception as e:
            print("Error during generation:", str(e))
            traceback.print_exc()
            raise  # don't swallow exceptions

    @observe(name="qwen_chat", as_type="generation")
    def chat(self, user_message, system_message="You are a helpful AI assistant."):
        prompt = f"System: {system_message}\nUser: {user_message}\nAssistant:"
        return self.generate_response(prompt, max_length=256)

@observe(name="main_conversation")
def run_conversation():
    chatbot = QwenChatbot()
    print("Chatbot initialized successfully!")

    examples = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms",
        "Write a short poem about artificial intelligence",
        "What are the benefits of renewable energy?"
    ]

    results = []
    for q in examples:
        print(f"\nQuestion: {q}")
        r = chatbot.chat(q)
        print(f"Response: {r}")
        results.append({"question": q, "response": r})
    return results

@observe(name="batch_processing")
def process_batch_questions(questions):
    chatbot = QwenChatbot()
    results = []
    for i, q in enumerate(questions, 1):
        print(f"Batch {i}/{len(questions)}: {q}")
        r = chatbot.generate_response(f"Q: {q}\nA:")
        results.append({"id": i, "question": q, "response": r})
    return results

@observe(name="performance_test")
def performance_test():
    chatbot = QwenChatbot()
    _ = chatbot.generate_response("Hello", max_length=50)
    start = time.time()
    response = chatbot.generate_response("Explain the concept of machine learning", max_length=200)
    duration = time.time() - start
    result = {"generation_time": duration, "response": response}

    if torch.cuda.is_available():
        result.update({
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1024**3,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1024**3
        })
        print(f"Generation time: {duration:.2f}s")
    return result

if __name__ == "__main__":
    run_conversation()
    print("\n=== PERFORMANCE TEST ===")
    print(performance_test())
    print("\n=== BATCH TEST ===")
    batch = ["What is Python?", "How does blockchain work?", "Explain neural networks"]
    process_batch_questions(batch)

    print("\nFlushing Langfuse observations...")
    try:
        langfuse.flush()
        print("✅ Successfully flushed to Langfuse!")
    except Exception as e:
        print("Error flushing to Langfuse:", e)
        _langfuse_enabled = False
