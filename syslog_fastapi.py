#!/usr/bin/env python3
"""
FastAPI Syslog ChromaDB API with Local LLM Chat Integration
Modern API with integrated web dashboard and Ollama chat for syslog analysis
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, List, Optional, Union, Any
import chromadb
import os
import ssl
import urllib3
import requests
import torch
from transformers import AutoTokenizer, AutoModel
import logging
from datetime import datetime, timedelta
import json
import traceback
from functools import lru_cache
import threading
import asyncio
import uvicorn
import httpx
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request/response
class SearchRequest(BaseModel):
    query: str
    limit: int = 10
    threshold: float = 0.0

class FilterRequest(BaseModel):
    filters: Dict
    limit: int = 100

class MessageRequest(BaseModel):
    message: str
    context: Optional[Dict] = None

class MessageResponse(BaseModel):
    success: bool
    response: str
    queries_executed: Optional[List[Dict]] = None
    logs_found: Optional[int] = None

class SearchResult(BaseModel):
    similarity: float
    timestamp: Optional[str]
    source_ip: Optional[str]
    facility: Optional[int]
    severity: Optional[int]
    message: str

class SearchResponse(BaseModel):
    success: bool
    query: Optional[str] = None
    results_count: int
    results: List[SearchResult]

class StatsResponse(BaseModel):
    success: bool
    data: Dict

# Initialize FastAPI
app = FastAPI(
    title="Syslog ChromaDB API with LLM Chat",
    description="AI-powered semantic search and analysis for syslog messages with natural language chat",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

class SyslogChatBot:
    """Local LLM chat interface for syslog analysis"""
    
    def __init__(self, ollama_host: str = "http://localhost:11434", model_name: str = "llama3.1:8b"):
        self.ollama_host = ollama_host
        self.model_name = model_name
        self.api = None  # Will be set later
        
    def set_api_reference(self, api_instance):
        """Set reference to main API for database queries"""
        self.api = api_instance
    
    async def _call_ollama(self, prompt: str) -> str:
        """Call Ollama API"""
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={
                        "model": self.model_name,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "temperature": 0.1,  # Low temperature for more consistent responses
                            "top_p": 0.9
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                    return "I'm having trouble connecting to the AI model. Please check if Ollama is running."
                    
        except httpx.TimeoutException:
            return "The AI model is taking too long to respond. Please try again."
        except Exception as e:
            logger.error(f"Error calling Ollama: {e}")
            return f"Error communicating with AI model: {str(e)}"
    
    async def _parse_query_intent_with_ollama(self, message: str) -> Dict:
        """Use Ollama to parse user intent and extract parameters"""
        
        intent_prompt = f"""Analyze this syslog query and extract structured information. Return ONLY a JSON object, no other text.

    Query: "{message}"

    Extract:
    1. intent: one of ["find_issues", "show_logs", "get_stats", "recent_logs", "general_search"]
    2. devices: list of device names, IPs, or hostnames mentioned
    3. time_refs: list of time references like ["1h", "1d", "7d"] 
    4. severity_keywords: list of severity levels (0-7: emergency, alert, critical, error, warning, notice, info, debug)
    5. search_terms: key terms for semantic search

    Rules:
    - "error", "fail", "down", "issue" → intent: "find_issues"
    - "show", "list", "get", "find" → intent: "show_logs"  
    - "stats", "count", "how many" → intent: "get_stats"
    - "recent", "latest", "last", "today" → intent: "recent_logs"
    - Extract device patterns: router-01, 192.168.1.1, rtr-core, switch-access
    - Map time words: "hour"→"1h", "day/today"→"1d", "week"→"7d"
    - Map severity: "emergency"→0, "alert"→1, "critical"→2, "error"→3, "warning"→4, "notice"→5, "info"→6, "debug"→7

    Return JSON format:
    {{"intent": "...", "devices": [...], "time_refs": [...], "severity_keywords": [...], "search_terms": [...]}}"""

        try:
            response = await self._call_ollama(intent_prompt)
            # Try to find JSON in response
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                intent_data = json.loads(json_match.group())
                intent_data["original_message"] = message
                return intent_data
            else:
                raise ValueError("No JSON found")
                        
        except Exception as e:
            logger.error(f"Ollama intent parsing failed: {e}")
            # Fallback to rule-based parsing
            return self._parse_query_intent_original(message)

    async def _execute_search_queries_enhanced(self, intent_data: Dict) -> List[Dict]:
        """Execute searches based on Ollama-parsed intent"""
        queries_executed = []
        all_results = []
        
        try:
            # Use extracted search_terms for more targeted queries
            search_terms = intent_data.get("search_terms", [intent_data["original_message"]])
            devices = intent_data.get("devices", [])
            
            if intent_data["intent"] == "find_issues":
                # Combine issue terms with devices
                for term in search_terms[:2]:
                    query = f"{term} " + " ".join(devices[:2])
                    results = await self.api.search_semantic(query.strip(), n_results=5, threshold=0.3)
                    queries_executed.append({"type": "semantic_search", "query": query, "results": len(results["documents"][0])})
                    all_results.extend(results["documents"][0])
            
            elif intent_data["intent"] == "show_logs" and devices:
                for device in devices[:2]:
                    results = await self.api.search_semantic(device, n_results=10, threshold=0.2)
                    queries_executed.append({"type": "semantic_search", "query": device, "results": len(results["documents"][0])})
                    all_results.extend(results["documents"][0])
            
            elif intent_data["intent"] == "get_stats":
                # Could trigger stats API instead of search
                stats = await self.api.get_stats()
                return queries_executed, [f"Database contains {stats['total_documents']} logs from {stats.get('unique_sources', 0)} sources"]
            
            else:
                # Use all search terms
                for term in search_terms[:3]:
                    results = await self.api.search_semantic(term, n_results=8, threshold=0.2)
                    queries_executed.append({"type": "semantic_search", "query": term, "results": len(results["documents"][0])})
                    all_results.extend(results["documents"][0])
            
            return queries_executed, all_results[:15]
        
        except Exception as e:
            logger.error(f"Error executing enhanced search queries: {e}")
            return queries_executed, []
    
    
    async def process_message(self, message: str) -> Dict:
        """Process natural language message and return response"""
        try:
            # Parse intent
            intent_data = await self._parse_query_intent_with_ollama(message)
            logger.info(f"Parsed intent: {intent_data}")
            
            # Execute database queries
            queries_executed, search_results = await self._execute_search_queries_enhanced(intent_data)
            
            # Build context for LLM
            context = {
                "user_question": message,
                "intent": intent_data["intent"],
                "devices_mentioned": intent_data["devices"],
                "queries_executed": queries_executed,
                "total_results_found": len(search_results),
                "sample_logs": search_results[:5]  # Send top 5 to LLM
            }
            
            # Create prompt for LLM
            llm_prompt = self._build_llm_prompt(context)
            
            # Get LLM response
            llm_response = await self._call_ollama(llm_prompt)
            
            return {
                "success": True,
                "response": llm_response,
                "queries_executed": queries_executed,
                "logs_found": len(search_results)
            }
            
        except Exception as e:
            logger.error(f"Error processing message: {e}")
            return {
                "success": False,
                "response": f"I encountered an error while processing your question: {str(e)}",
                "queries_executed": [],
                "logs_found": 0
            }
    
    def _build_llm_prompt(self, context: Dict) -> str:
        """Build prompt for LLM based on context and search results"""
        
        sample_logs_text = ""
        if context["sample_logs"]:
            sample_logs_text = "\n".join([f"- {log}" for log in context["sample_logs"][:5]])
        else:
            sample_logs_text = "No relevant logs found."
        
        prompt = f"""You are a network security and log analysis expert. A user asked: "{context['user_question']}"

I searched the syslog database and found {context['total_results_found']} relevant log entries.

Here are the most relevant logs:
{sample_logs_text}

Database queries executed:
{json.dumps(context['queries_executed'], indent=2)}

Please analyze these logs and provide a helpful, concise response to the user's question. Focus on:
1. Direct answer to their question
2. Any issues or patterns you identify
3. Specific details from the logs that are relevant
4. If no relevant logs were found, suggest alternative search terms or indicate this

Keep your response focused and actionable. If you see error patterns or issues, mention them specifically."""

        return prompt

class SyslogAPI:
    """API interface for syslog ChromaDB queries"""
    
    def __init__(self, db_path: str = "/var/syslog_chromadb", model_name: str = "mixedbread-ai/mxbai-embed-large-v1"):
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.client = None
        self.collection = None
        self._model_lock = threading.Lock()
        
        # Setup environment
        self._setup_environment()
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Initialize model
        self._init_model()
    
    def _setup_environment(self):
        """Setup environment for corporate network"""
        os.environ['HF_HOME'] = '/var/cache/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/var/cache/huggingface'
        
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings()
        
        original_send = requests.adapters.HTTPAdapter.send
        def bypass_ssl_send(self, request, *args, **kwargs):
            kwargs['verify'] = False
            return original_send(self, request, *args, **kwargs)
        requests.adapters.HTTPAdapter.send = bypass_ssl_send
    
    def _init_chromadb(self):
        """Initialize ChromaDB connection"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_collection("syslog_messages")
            logger.info(f"Connected to ChromaDB: {self.collection.count():,} documents")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def _init_model(self):
        """Initialize embedding model"""
        try:
            with self._model_lock:
                if self.model is None:
                    logger.info(f"Loading embedding model: {self.model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @lru_cache(maxsize=100)
    def _encode_query(self, query_text: str):
        """Encode query text to embedding vector (cached)"""
        if not self.model or not self.tokenizer:
            raise Exception("Model not initialized")
        
        with torch.no_grad():
            inputs = self.tokenizer(
                query_text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()
    
    def _get_database_size(self) -> float:
        """Get database size in GB"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.db_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024**3)  # Convert to GB
        except Exception:
            return 0.0
    
    async def search_semantic(self, query: str, n_results: int = 10, threshold: float = 0.0) -> Dict:
        """Perform semantic search"""
        try:
            query_embedding = self._encode_query(query)
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Filter by similarity threshold
            filtered_results = {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                similarity = 1 - dist
                if similarity >= threshold:
                    filtered_results['documents'][0].append(doc)
                    filtered_results['metadatas'][0].append(meta)
                    filtered_results['distances'][0].append(dist)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    async def search_filter(self, filters: Dict, n_results: int = 100) -> Dict:
        """Search using metadata filters"""
        try:
            dummy_embedding = self._encode_query("filter")
            
            results = self.collection.query(
                query_embeddings=dummy_embedding.tolist(),
                n_results=n_results,
                where=filters,
                include=['documents', 'metadatas']
            )
            return results
        except Exception as e:
            logger.error(f"Filter search error: {e}")
            raise
    
    async def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            total_count = self.collection.count()
            
            stats = {
                'total_documents': total_count,
                'database_path': self.db_path,
                'database_size_gb': round(self._get_database_size(), 2),
                'model': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            if total_count > 0:
                sample_size = min(1000, total_count)
                dummy_embedding = self._encode_query("sample")
                
                sample_results = self.collection.query(
                    query_embeddings=dummy_embedding.tolist(),
                    n_results=sample_size,
                    include=['metadatas']
                )
                
                stats['sample_size'] = sample_size
                
                if sample_results['metadatas'] and sample_results['metadatas'][0]:
                    metadatas = sample_results['metadatas'][0]
                    
                    facilities = {}
                    severities = {}
                    sources = set()
                    timestamps = []
                    
                    for metadata in metadatas:
                        if metadata.get('facility') is not None:
                            facilities[metadata['facility']] = facilities.get(metadata['facility'], 0) + 1
                        if metadata.get('severity') is not None:
                            severities[metadata['severity']] = severities.get(metadata['severity'], 0) + 1
                        if metadata.get('source_ip'):
                            sources.add(metadata['source_ip'])
                        if metadata.get('timestamp'):
                            timestamps.append(metadata['timestamp'])
                    
                    stats.update({
                        'facilities': facilities,
                        'severities': severities,
                        'unique_sources': len(sources),
                        'top_sources': list(sources)[:10]
                    })
                    
                    if timestamps:
                        timestamps.sort()
                        stats['time_range'] = {
                            'earliest': timestamps[0],
                            'latest': timestamps[-1]
                        }
            else:
                stats['sample_size'] = 0
                stats['message'] = 'No documents in database yet'
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            raise
    """API interface for syslog ChromaDB queries"""
    
    def __init__(self, db_path: str = "/var/syslog_chromadb", model_name: str = "mixedbread-ai/mxbai-embed-large-v1"):
        self.db_path = db_path
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.device = None
        self.client = None
        self.collection = None
        self._model_lock = threading.Lock()
        
        # Setup environment
        self._setup_environment()
        
        # Initialize ChromaDB
        self._init_chromadb()
        
        # Initialize model
        self._init_model()
    
    def _setup_environment(self):
        """Setup environment for corporate network"""
        os.environ['HF_HOME'] = '/var/cache/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/var/cache/huggingface'
        
        ssl._create_default_https_context = ssl._create_unverified_context
        urllib3.disable_warnings()
        
        original_send = requests.adapters.HTTPAdapter.send
        def bypass_ssl_send(self, request, *args, **kwargs):
            kwargs['verify'] = False
            return original_send(self, request, *args, **kwargs)
        requests.adapters.HTTPAdapter.send = bypass_ssl_send
    
    def _init_chromadb(self):
        """Initialize ChromaDB connection"""
        try:
            self.client = chromadb.PersistentClient(path=self.db_path)
            self.collection = self.client.get_collection("syslog_messages")
            logger.info(f"Connected to ChromaDB: {self.collection.count():,} documents")
        except Exception as e:
            logger.error(f"Failed to connect to ChromaDB: {e}")
            raise
    
    def _init_model(self):
        """Initialize embedding model"""
        try:
            with self._model_lock:
                if self.model is None:
                    logger.info(f"Loading embedding model: {self.model_name}")
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                    self.model = AutoModel.from_pretrained(self.model_name)
                    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                    self.model.to(self.device)
                    self.model.eval()
                    logger.info(f"Model loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    @lru_cache(maxsize=100)
    def _encode_query(self, query_text: str):
        """Encode query text to embedding vector (cached)"""
        if not self.model or not self.tokenizer:
            raise Exception("Model not initialized")
        
        with torch.no_grad():
            inputs = self.tokenizer(
                query_text,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            # Mean pooling
            attention_mask = inputs['attention_mask']
            token_embeddings = outputs.last_hidden_state
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            return embeddings.cpu().numpy()
    
    def _get_database_size(self) -> float:
        """Get database size in GB"""
        try:
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.db_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size / (1024**3)  # Convert to GB
        except Exception:
            return 0.0
    
    async def search_semantic(self, query: str, n_results: int = 10, threshold: float = 0.0) -> Dict:
        """Perform semantic search"""
        try:
            query_embedding = self._encode_query(query)
            
            results = self.collection.query(
                query_embeddings=query_embedding.tolist(),
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            # Filter by similarity threshold
            filtered_results = {
                'documents': [[]],
                'metadatas': [[]],
                'distances': [[]]
            }
            
            for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
                similarity = 1 - dist
                if similarity >= threshold:
                    filtered_results['documents'][0].append(doc)
                    filtered_results['metadatas'][0].append(meta)
                    filtered_results['distances'][0].append(dist)
            
            return filtered_results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            raise
    
    async def search_filter(self, filters: Dict, n_results: int = 100) -> Dict:
        """Search using metadata filters"""
        try:
            dummy_embedding = self._encode_query("filter")
            
            results = self.collection.query(
                query_embeddings=dummy_embedding.tolist(),
                n_results=n_results,
                where=filters,
                include=['documents', 'metadatas']
            )
            return results
        except Exception as e:
            logger.error(f"Filter search error: {e}")
            raise
    
    async def get_stats(self) -> Dict:
        """Get database statistics"""
        try:
            total_count = self.collection.count()
            
            stats = {
                'total_documents': total_count,
                'database_path': self.db_path,
                'database_size_gb': round(self._get_database_size(), 2),
                'model': self.model_name,
                'timestamp': datetime.now().isoformat()
            }
            
            if total_count > 0:
                sample_size = min(1000, total_count)
                dummy_embedding = self._encode_query("sample")
                
                sample_results = self.collection.query(
                    query_embeddings=dummy_embedding.tolist(),
                    n_results=sample_size,
                    include=['metadatas']
                )
                
                stats['sample_size'] = sample_size
                
                if sample_results['metadatas'] and sample_results['metadatas'][0]:
                    metadatas = sample_results['metadatas'][0]
                    
                    facilities = {}
                    severities = {}
                    sources = set()
                    timestamps = []
                    
                    for metadata in metadatas:
                        if metadata.get('facility') is not None:
                            facilities[metadata['facility']] = facilities.get(metadata['facility'], 0) + 1
                        if metadata.get('severity') is not None:
                            severities[metadata['severity']] = severities.get(metadata['severity'], 0) + 1
                        if metadata.get('source_ip'):
                            sources.add(metadata['source_ip'])
                        if metadata.get('timestamp'):
                            timestamps.append(metadata['timestamp'])
                    
                    stats.update({
                        'facilities': facilities,
                        'severities': severities,
                        'unique_sources': len(sources),
                        'top_sources': list(sources)[:10]
                    })
                    
                    if timestamps:
                        timestamps.sort()
                        stats['time_range'] = {
                            'earliest': timestamps[0],
                            'latest': timestamps[-1]
                        }
            else:
                stats['sample_size'] = 0
                stats['message'] = 'No documents in database yet'
            
            return stats
            
        except Exception as e:
            logger.error(f"Stats error: {e}")
            raise

# Initialize API and ChatBot
api = SyslogAPI()
chatbot = SyslogChatBot()
chatbot.set_api_reference(api)

# Dashboard HTML template
dashboard_html = """
<!DOCTYPE html>
<html>
<head>
    <title>Syslog Analytics Dashboard</title>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #f5f7fa;
            color: #2d3748;
        }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        .header { 
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; 
            padding: 30px 0; 
            margin: -20px -20px 30px -20px;
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .header h1 { font-size: 2.5rem; margin-bottom: 10px; }
        .header p { font-size: 1.1rem; opacity: 0.9; }
        .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .card { 
            background: white; 
            border-radius: 12px; 
            padding: 25px; 
            box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            border: 1px solid #e2e8f0;
        }
        .card h3 { 
            color: #2d3748; 
            margin-bottom: 15px; 
            font-size: 1.3rem;
            display: flex;
            align-items: center;
        }
        .icon { 
            width: 24px; 
            height: 24px; 
            margin-right: 10px;
            border-radius: 6px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 14px;
        }
        .stat-value { 
            font-size: 2.5rem; 
            font-weight: bold; 
            color: #667eea;
            margin: 10px 0;
        }
        .stat-label { 
            color: #718096; 
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .search-box { 
            margin-bottom: 30px;
            display: flex;
            gap: 10px;
        }
        .search-box input { 
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #e2e8f0;
            border-radius: 8px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.2s;
        }
        .search-box input:focus { border-color: #667eea; }
        .search-box button { 
            padding: 12px 24px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 500;
            transition: background 0.2s;
        }
        .search-box button:hover { background: #5a67d8; }
        .chat-button { background: #38b2ac !important; }
        .chat-button:hover { background: #319795 !important; }
        .results { margin-top: 20px; }
        .result-item { 
            background: white;
            padding: 15px;
            margin-bottom: 10px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }
        .result-meta { 
            font-size: 0.85rem;
            color: #718096;
            margin-bottom: 8px;
        }
        .result-message { 
            font-family: 'Monaco', 'Menlo', monospace;
            font-size: 0.9rem;
            background: #f7fafc;
            padding: 8px;
            border-radius: 4px;
        }
        .chat-response {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            margin-bottom: 15px;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .chat-response h4 {
            margin: 0 0 10px 0;
            font-size: 1.1rem;
            opacity: 0.9;
        }
        .chat-response .response-text {
            font-size: 1rem;
            line-height: 1.6;
            margin-bottom: 15px;
        }
        .chat-response .meta-info {
            font-size: 0.85rem;
            opacity: 0.8;
            border-top: 1px solid rgba(255,255,255,0.2);
            padding-top: 10px;
        }
        .chart-container { height: 300px; position: relative; }
        .refresh-btn {
            position: fixed;
            bottom: 20px;
            right: 20px;
            background: #667eea;
            color: white;
            border: none;
            border-radius: 50%;
            width: 60px;
            height: 60px;
            font-size: 24px;
            cursor: pointer;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            transition: all 0.2s;
        }
        .refresh-btn:hover { transform: scale(1.1); }
        .loading { 
            display: none;
            text-align: center;
            padding: 20px;
            color: #718096;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔍 Syslog Analytics Dashboard</h1>
            <p>AI-powered semantic search and real-time monitoring with natural language chat</p>
        </div>

        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search logs semantically (e.g., 'BGP neighbor down', 'authentication failed')">
            <button onclick="searchLogs()">Search</button>
        </div>

        <div class="search-box">
            <input type="text" id="chatInput" placeholder="Ask a question about your logs (e.g., 'Are there issues with rtr-01?', 'What happened in the last hour?')">
            <button onclick="askQuestion()" style="background: #38b2ac;">Ask AI</button>
        </div>

        <div class="grid">
            <div class="card">
                <h3><span class="icon" style="background: #48bb78; color: white;">📊</span>Database Stats</h3>
                <div class="stat-value" id="totalDocs">-</div>
                <div class="stat-label">Total Documents</div>
                <div style="margin-top: 15px;">
                    <div style="font-size: 1rem; font-weight: 500; color: #2d3748; margin-bottom: 8px;">Top 5 Sources by Message Count:</div>
                    <div style="font-size: 1.2rem; color: #2d3748;" id="dbSize">- GB</div>
                    <div class="stat-label">Database Size</div>
                </div>
            </div>

            <div class="card">
                <h3><span class="icon" style="background: #ed8936; color: white;">🌐</span>Sources</h3>
                <div class="stat-value" id="uniqueSources">-</div>
                <div class="stat-label">Unique Sources</div>
                <div style="margin-top: 15px;" id="topSources">Loading...</div>
            </div>

            <div class="card">
                <h3><span class="icon" style="background: #9f7aea; color: white;">⚠️</span>Severity Distribution</h3>
                <div class="chart-container">
                    <canvas id="severityChart"></canvas>
                </div>
            </div>
        </div>

        <div class="results" id="searchResults"></div>

        <div class="loading" id="loading">
            <div>⏳ Loading...</div>
        </div>
    </div>

    <button class="refresh-btn" onclick="loadStats()" title="Refresh Data">🔄</button>

    <script>
        let severityChart;

        // Initialize charts
        function initCharts() {
            const severityCtx = document.getElementById('severityChart').getContext('2d');

            severityChart = new Chart(severityCtx, {
                type: 'doughnut',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom' } }
                }
            });
        }

        // Load statistics
        async function loadStats() {
            try {
                const response = await fetch('/api/stats');
                const data = await response.json();
                
                if (data.success) {
                    const stats = data.data;
                    
                    // Update stats
                    document.getElementById('totalDocs').textContent = stats.total_documents.toLocaleString();
                    document.getElementById('dbSize').textContent = `${stats.database_size_gb} GB`;
                    document.getElementById('uniqueSources').textContent = stats.unique_sources || 0;
                    
                    // Update top sources
                    if (stats.top_sources) {
                        const topSourcesHtml = stats.top_sources.slice(0, 5).map((ip, index) => 
                            `<div style="font-size: 0.85rem; margin: 3px 0; padding: 2px 0;">
                                <span style="font-weight: 500;">${index + 1}.</span> ${ip}
                            </div>`
                        ).join('');
                        document.getElementById('topSources').innerHTML = topSourcesHtml;
                    }
                    
                    // Update severity chart
                    if (stats.severities) {
                        const severityNames = {0: 'Emergency', 1: 'Alert', 2: 'Critical', 3: 'Error', 4: 'Warning', 5: 'Notice', 6: 'Info', 7: 'Debug'};
                        const severityData = Object.entries(stats.severities).map(([k, v]) => ({
                            label: severityNames[k] || `Level ${k}`,
                            value: v
                        }));
                        
                        severityChart.data = {
                            labels: severityData.map(d => d.label),
                            datasets: [{
                                data: severityData.map(d => d.value),
                                backgroundColor: ['#f56565', '#ed8936', '#ecc94b', '#48bb78', '#38b2ac', '#4299e1', '#9f7aea', '#ed64a6']
                            }]
                        };
                        severityChart.update();
                    }
                }
            } catch (error) {
                console.error('Error loading stats:', error);
            }
        }

        // Ask AI question
        async function askQuestion() {
            const question = document.getElementById('chatInput').value.trim();
            if (!question) return;

            const resultsDiv = document.getElementById('searchResults');
            const loading = document.getElementById('loading');
            
            loading.style.display = 'block';
            resultsDiv.innerHTML = '';

            try {
                const response = await fetch('/api/message', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ message: question })
                });
                
                const data = await response.json();
                loading.style.display = 'none';
                
                if (data.success) {
                    resultsDiv.innerHTML = `
                        <div class="card">
                            <div class="chat-response">
                                <h4>🤖 AI Analysis</h4>
                                <div class="response-text">${data.response}</div>
                                <div class="meta-info">
                                    Found ${data.logs_found || 0} relevant logs • 
                                    Executed ${data.queries_executed?.length || 0} queries
                                </div>
                            </div>
                            ${data.queries_executed && data.queries_executed.length > 0 ? `
                                <h4>🔍 Queries Executed</h4>
                                ${data.queries_executed.map(query => `
                                    <div style="background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 5px; font-size: 0.9rem;">
                                        <strong>${query.type}:</strong> ${query.query || query.filter} 
                                        <span style="color: #666;">(${query.results} results)</span>
                                    </div>
                                `).join('')}
                            ` : ''}
                        </div>
                    `;
                } else {
                    resultsDiv.innerHTML = `
                        <div class="card">
                            <h3>🤖 AI Response</h3>
                            <p style="color: #e53e3e;">Error: ${data.response || 'Failed to get AI response'}</p>
                        </div>
                    `;
                }
            } catch (error) {
                loading.style.display = 'none';
                resultsDiv.innerHTML = `
                    <div class="card">
                        <h3>🤖 AI Response</h3>
                        <p style="color: #e53e3e;">Error: ${error.message}</p>
                        <p style="font-size: 0.9rem; color: #666;">Make sure Ollama is running and a model is installed.</p>
                    </div>
                `;
                console.error('Chat error:', error);
            }
        }

        // Search logs
        async function searchLogs() {
            const query = document.getElementById('searchInput').value.trim();
            if (!query) return;

            const resultsDiv = document.getElementById('searchResults');
            const loading = document.getElementById('loading');
            
            loading.style.display = 'block';
            resultsDiv.innerHTML = '';

            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, limit: 10 })
                });
                
                const data = await response.json();
                loading.style.display = 'none';
                
                if (data.success && data.results.length > 0) {
                    resultsDiv.innerHTML = `
                        <div class="card">
                            <h3>🔍 Search Results for "${query}" (${data.results_count} found)</h3>
                            ${data.results.map(result => `
                                <div class="result-item">
                                    <div class="result-meta">
                                        Similarity: ${(result.similarity * 100).toFixed(1)}% | 
                                        Source: ${result.source_ip} | 
                                        Time: ${new Date(result.timestamp).toLocaleString()} |
                                        Severity: ${result.severity}
                                    </div>
                                    <div class="result-message">${result.message}</div>
                                </div>
                            `).join('')}
                        </div>
                    `;
                } else {
                    resultsDiv.innerHTML = `
                        <div class="card">
                            <h3>🔍 Search Results</h3>
                            <p>No results found for "${query}"</p>
                        </div>
                    `;
                }
            } catch (error) {
                loading.style.display = 'none';
                console.error('Search error:', error);
            }
        }

        // Handle Enter key in search and chat inputs
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') searchLogs();
        });
        
        document.getElementById('chatInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') askQuestion();
        });

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {
            initCharts();
            loadStats();
            setInterval(loadStats, 30000); // Refresh every 30 seconds
        });
    </script>
</body>
</html>
"""

# API Routes
@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page"""
    return dashboard_html

@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get database statistics"""
    try:
        stats = await api.get_stats()
        return StatsResponse(success=True, data=stats)
    except Exception as e:
        logger.error(f"Stats error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """Semantic search through syslog messages"""
    try:
        if not request.query:
            raise HTTPException(status_code=400, detail="Query parameter is required")
        
        results = await api.search_semantic(request.query, n_results=request.limit, threshold=request.threshold)
        
        # Format results
        formatted_results = []
        for doc, meta, dist in zip(results['documents'][0], results['metadatas'][0], results['distances'][0]):
            formatted_results.append(SearchResult(
                similarity=round(1 - dist, 3),
                timestamp=meta.get('timestamp'),
                source_ip=meta.get('source_ip'),
                facility=meta.get('facility'),
                severity=meta.get('severity'),
                message=doc
            ))
        
        return SearchResponse(
            success=True,
            query=request.query,
            results_count=len(formatted_results),
            results=formatted_results
        )
        
    except Exception as e:
        logger.error(f"Search error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/message", response_model=MessageResponse)
async def chat_message(request: MessageRequest):
    """Natural language chat interface for syslog analysis"""
    try:
        if not request.message.strip():
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        
        result = await chatbot.process_message(request.message)
        
        return MessageResponse(
            success=result["success"],
            response=result["response"],
            queries_executed=result.get("queries_executed"),
            logs_found=result.get("logs_found")
        )
        
    except Exception as e:
        logger.error(f"Chat message error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/message")
async def chat_message_get(q: str):
    """GET-based chat interface for simple queries"""
    request = MessageRequest(message=q)
    return await chat_message(request)

async def search_get(q: str, limit: int = 10, threshold: float = 0.0):
    """GET-based search for simple queries"""
    request = SearchRequest(query=q, limit=limit, threshold=threshold)
    return await search(request)

@app.post("/api/filter")
async def filter_search(request: FilterRequest):
    """Filter messages by metadata"""
    try:
        if not request.filters:
            raise HTTPException(status_code=400, detail="Filters are required")
        
        results = await api.search_filter(request.filters, n_results=request.limit)
        
        # Format results
        formatted_results = []
        for doc, meta in zip(results['documents'][0], results['metadatas'][0]):
            formatted_results.append({
                'timestamp': meta.get('timestamp'),
                'source_ip': meta.get('source_ip'),
                'facility': meta.get('facility'),
                'severity': meta.get('severity'),
                'message': doc
            })
        
        return {
            'success': True,
            'filters': request.filters,
            'results_count': len(formatted_results),
            'results': formatted_results
        }
        
    except Exception as e:
        logger.error(f"Filter error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        count = api.collection.count()
        return {
            'success': True,
            'status': 'healthy',
            'documents': count,
            'timestamp': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }

if __name__ == '__main__':
    print("🚀 Starting Syslog ChromaDB FastAPI with LLM Chat")
    print(f"📊 Database: {api.db_path}")
    print(f"📄 Documents: {api.collection.count():,}")
    print(f"🔥 Model: {api.model_name}")
    print(f"🤖 Chat LLM: {chatbot.model_name}")
    print("🌐 Dashboard: http://localhost:8000")
    print("📋 API Docs: http://localhost:8000/api/docs")
    print("💬 Chat: POST /api/message")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
