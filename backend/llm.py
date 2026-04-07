"""LLM integration for Toronto Bylaw Agent."""
import aiohttp
import json
from typing import List, Dict, Any
from backend.config import LLM_API_URL, LLM_API_KEY, LLM_MODEL


class LLMClient:
    """Client for interacting with LLM API."""
    
    def __init__(self):
        """Initialize LLM client."""
        self.api_url = LLM_API_URL
        self.api_key = LLM_API_KEY
        self.model = LLM_MODEL
    
    async def invoke(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> str:
        """Invoke LLM with messages."""
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise Exception(f"LLM API error: {response.status} - {error_text}")
                
                result = await response.json()
                return result["choices"][0]["message"]["content"]


# Global LLM client
llm_client = LLMClient()
