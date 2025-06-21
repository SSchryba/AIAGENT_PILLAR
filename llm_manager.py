#!/usr/bin/env python3
"""
Multi-LLM Manager
Supports multiple Hugging Face models with automatic switching and optimization.
"""

import os
import logging
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from enum import Enum
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM,
    pipeline, BitsAndBytesConfig, GenerationConfig
)
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms import HuggingFacePipeline
import agent_config as config

logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types."""
    CAUSAL_LM = "causal_lm"
    SEQ2SEQ = "seq2seq"
    CHAT = "chat"

@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    name: str
    model_id: str
    model_type: ModelType
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.1
    do_sample: bool = True
    use_quantization: bool = True
    device_map: str = "auto"
    trust_remote_code: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = True

class MultiLLMManager:
    """Manages multiple LLM models with automatic switching and optimization."""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.tokenizers: Dict[str, Any] = {}
        self.pipelines: Dict[str, Any] = {}
        self.current_model: Optional[str] = None
        self.model_configs: Dict[str, ModelConfig] = {}
        self.performance_metrics: Dict[str, Dict[str, float]] = {}
        
        # Initialize default models
        self._initialize_default_models()
    
    def _initialize_default_models(self):
        """Initialize default model configurations."""
        default_models = [
            ModelConfig(
                name="llama2-7b",
                model_id="meta-llama/Llama-2-7b-chat-hf",
                model_type=ModelType.CHAT,
                max_length=4096,
                temperature=0.7
            ),
            ModelConfig(
                name="mistral-7b",
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
                model_type=ModelType.CHAT,
                max_length=4096,
                temperature=0.7
            ),
            ModelConfig(
                name="codellama-7b",
                model_id="codellama/CodeLlama-7b-Instruct-hf",
                model_type=ModelType.CHAT,
                max_length=4096,
                temperature=0.7
            ),
            ModelConfig(
                name="phi-2",
                model_id="microsoft/phi-2",
                model_type=ModelType.CAUSAL_LM,
                max_length=2048,
                temperature=0.7
            ),
            ModelConfig(
                name="flan-t5-small",
                model_id="google/flan-t5-small",
                model_type=ModelType.SEQ2SEQ,
                max_length=512,
                temperature=0.7
            )
        ]
        
        for model_config in default_models:
            self.model_configs[model_config.name] = model_config
    
    def _get_quantization_config(self, model_config: ModelConfig) -> Optional[BitsAndBytesConfig]:
        """Get quantization configuration for a model."""
        if not model_config.use_quantization:
            return None
        
        if model_config.load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
        elif model_config.load_in_8bit:
            return BitsAndBytesConfig(load_in_8bit=True)
        
        return None
    
    def _get_generation_config(self, model_config: ModelConfig) -> GenerationConfig:
        """Get generation configuration for a model."""
        return GenerationConfig(
            max_length=model_config.max_length,
            temperature=model_config.temperature,
            top_p=model_config.top_p,
            top_k=model_config.top_k,
            repetition_penalty=model_config.repetition_penalty,
            do_sample=model_config.do_sample,
            pad_token_id=0,
            eos_token_id=2
        )
    
    def load_model(self, model_name: str, force_reload: bool = False) -> bool:
        """Load a specific model."""
        if model_name in self.models and not force_reload:
            logger.info(f"Model {model_name} already loaded")
            return True
        
        if model_name not in self.model_configs:
            logger.error(f"Model {model_name} not found in configurations")
            return False
        
        model_config = self.model_configs[model_name]
        
        try:
            logger.info(f"Loading model: {model_name} ({model_config.model_id})")
            start_time = time.time()
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_config.model_id,
                trust_remote_code=model_config.trust_remote_code
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Get quantization config
            quantization_config = self._get_quantization_config(model_config)
            
            # Load model based on type
            if model_config.model_type == ModelType.CAUSAL_LM:
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_id,
                    quantization_config=quantization_config,
                    device_map=model_config.device_map,
                    trust_remote_code=model_config.trust_remote_code,
                    torch_dtype=torch.float16
                )
            elif model_config.model_type == ModelType.SEQ2SEQ:
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_config.model_id,
                    quantization_config=quantization_config,
                    device_map=model_config.device_map,
                    trust_remote_code=model_config.trust_remote_code,
                    torch_dtype=torch.float16
                )
            else:  # CHAT model
                model = AutoModelForCausalLM.from_pretrained(
                    model_config.model_id,
                    quantization_config=quantization_config,
                    device_map=model_config.device_map,
                    trust_remote_code=model_config.trust_remote_code,
                    torch_dtype=torch.float16
                )
            
            # Create pipeline
            generation_config = self._get_generation_config(model_config)
            
            pipe = pipeline(
                "text-generation" if model_config.model_type != ModelType.SEQ2SEQ else "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                generation_config=generation_config,
                device_map=model_config.device_map
            )
            
            # Store components
            self.models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.pipelines[model_name] = pipe
            
            # Initialize performance metrics
            self.performance_metrics[model_name] = {
                "load_time": time.time() - start_time,
                "total_requests": 0,
                "total_tokens": 0,
                "average_response_time": 0.0,
                "error_count": 0
            }
            
            logger.info(f"Model {model_name} loaded successfully in {time.time() - start_time:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return False
    
    def unload_model(self, model_name: str) -> bool:
        """Unload a specific model to free memory."""
        try:
            if model_name in self.models:
                del self.models[model_name]
                del self.tokenizers[model_name]
                del self.pipelines[model_name]
                
                # Clear CUDA cache if using GPU
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Model {model_name} unloaded successfully")
                return True
            return False
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    def switch_model(self, model_name: str) -> bool:
        """Switch to a different model."""
        if model_name not in self.model_configs:
            logger.error(f"Model {model_name} not found in configurations")
            return False
        
        if self.load_model(model_name):
            self.current_model = model_name
            logger.info(f"Switched to model: {model_name}")
            return True
        return False
    
    def get_current_model(self) -> Optional[str]:
        """Get the currently active model."""
        return self.current_model
    
    def get_available_models(self) -> List[str]:
        """Get list of available model names."""
        return list(self.model_configs.keys())
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        return list(self.models.keys())
    
    def predict(self, prompt: str, model_name: Optional[str] = None) -> str:
        """Generate text using the specified or current model."""
        if model_name is None:
            model_name = self.current_model
        
        if model_name is None:
            # Load default model if none is set
            default_model = list(self.model_configs.keys())[0]
            self.switch_model(default_model)
            model_name = default_model
        
        if model_name not in self.pipelines:
            logger.error(f"Model {model_name} not loaded")
            return "Error: Model not available"
        
        try:
            start_time = time.time()
            
            # Format prompt based on model type
            model_config = self.model_configs[model_name]
            formatted_prompt = self._format_prompt(prompt, model_config)
            
            # Generate response
            response = self.pipelines[model_name](
                formatted_prompt,
                max_new_tokens=model_config.max_length,
                do_sample=model_config.do_sample,
                temperature=model_config.temperature,
                top_p=model_config.top_p,
                top_k=model_config.top_k,
                repetition_penalty=model_config.repetition_penalty,
                pad_token_id=self.tokenizers[model_name].pad_token_id,
                eos_token_id=self.tokenizers[model_name].eos_token_id
            )
            
            # Extract generated text
            if model_config.model_type == ModelType.SEQ2SEQ:
                generated_text = response[0]['generated_text']
            else:
                # Remove the input prompt from the response
                full_text = response[0]['generated_text']
                generated_text = full_text[len(formatted_prompt):].strip()
            
            # Update performance metrics
            response_time = time.time() - start_time
            self._update_metrics(model_name, response_time, len(generated_text.split()))
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error generating text with model {model_name}: {e}")
            self.performance_metrics[model_name]["error_count"] += 1
            return f"Error: {str(e)}"
    
    def _format_prompt(self, prompt: str, model_config: ModelConfig) -> str:
        """Format prompt based on model type and requirements."""
        if model_config.model_type == ModelType.CHAT:
            # Format for chat models
            if "llama" in model_config.model_id.lower():
                return f"<s>[INST] {prompt} [/INST]"
            elif "mistral" in model_config.model_id.lower():
                return f"<s>[INST] {prompt} [/INST]"
            elif "codellama" in model_config.model_id.lower():
                return f"[INST] {prompt} [/INST]"
            else:
                return f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        else:
            # For causal LM and seq2seq models
            return prompt
    
    def _update_metrics(self, model_name: str, response_time: float, token_count: int):
        """Update performance metrics for a model."""
        metrics = self.performance_metrics[model_name]
        metrics["total_requests"] += 1
        metrics["total_tokens"] += token_count
        
        # Update average response time
        current_avg = metrics["average_response_time"]
        total_requests = metrics["total_requests"]
        metrics["average_response_time"] = (current_avg * (total_requests - 1) + response_time) / total_requests
    
    def get_model_performance(self, model_name: str) -> Dict[str, Any]:
        """Get performance metrics for a specific model."""
        if model_name not in self.performance_metrics:
            return {}
        
        metrics = self.performance_metrics[model_name].copy()
        metrics["model_config"] = self.model_configs[model_name].__dict__
        metrics["is_loaded"] = model_name in self.models
        metrics["is_current"] = model_name == self.current_model
        
        return metrics
    
    def get_all_performance_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all models."""
        return {
            model_name: self.get_model_performance(model_name)
            for model_name in self.model_configs.keys()
        }
    
    def add_custom_model(self, model_config: ModelConfig) -> bool:
        """Add a custom model configuration."""
        try:
            self.model_configs[model_config.name] = model_config
            logger.info(f"Added custom model configuration: {model_config.name}")
            return True
        except Exception as e:
            logger.error(f"Failed to add custom model {model_config.name}: {e}")
            return False
    
    def optimize_memory(self):
        """Optimize memory usage by unloading unused models."""
        loaded_models = self.get_loaded_models()
        if len(loaded_models) > 2:  # Keep only current model and one backup
            for model_name in loaded_models:
                if model_name != self.current_model:
                    self.unload_model(model_name)
                    break

# Global LLM manager instance
_llm_manager = None

def get_llm_manager() -> MultiLLMManager:
    """Get the global LLM manager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = MultiLLMManager()
    return _llm_manager

class HuggingFaceLLM(LLM):
    """LangChain-compatible LLM wrapper for Hugging Face models."""
    
    def __init__(self, model_name: str = "mistral-7b"):
        super().__init__()
        self.model_name = model_name
        self.llm_manager = get_llm_manager()
        
        # Load the model
        if not self.llm_manager.load_model(model_name):
            raise ValueError(f"Failed to load model: {model_name}")
        
        self.llm_manager.switch_model(model_name)
    
    @property
    def _llm_type(self) -> str:
        return "huggingface"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Generate text using the Hugging Face model."""
        return self.llm_manager.predict(prompt, self.model_name)
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get identifying parameters."""
        return {
            "model_name": self.model_name,
            "model_type": "huggingface"
        } 