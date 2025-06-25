#!/usr/bin/env python3
"""
FastAPI Syslog ChromaDB API with Real-time Dashboard
Modern API with integrated web dashboard for syslog analysis
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Dict, List, Optional, Union
import chromadb
import os
import ssl
import urllib3
import requests
import torch
from transformers import AutoTokenizer, AutoModel
import logging
from datetime import datetime
import json
import traceback
from functools import lru_cache
import threading
import asyncio
import uvicorn

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
    title="Syslog ChromaDB API",
    description="AI-powered semantic search and analysis for syslog messages",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc"
)

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

# Initialize API
api = SyslogAPI()

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
            <p>AI-powered semantic search and real-time monitoring</p>
        </div>

        <div class="search-box">
            <input type="text" id="searchInput" placeholder="Search logs semantically (e.g., 'BGP neighbor down', 'authentication failed')">
            <button onclick="searchLogs()">Search</button>
        </div>

        <div class="grid">
            <div class="card">
                <h3><span class="icon" style="background: #48bb78; color: white;">📊</span>Database Stats</h3>
                <div class="stat-value" id="totalDocs">-</div>
                <div class="stat-label">Total Documents</div>
                <div style="margin-top: 15px;">
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

            <div class="card">
                <h3><span class="icon" style="background: #38b2ac; color: white;">🔧</span>Facility Distribution</h3>
                <div class="chart-container">
                    <canvas id="facilityChart"></canvas>
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
        let severityChart, facilityChart;

        // Initialize charts
        function initCharts() {
            const severityCtx = document.getElementById('severityChart').getContext('2d');
            const facilityCtx = document.getElementById('facilityChart').getContext('2d');

            severityChart = new Chart(severityCtx, {
                type: 'doughnut',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { position: 'bottom' } }
                }
            });

            facilityChart = new Chart(facilityCtx, {
                type: 'bar',
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: { legend: { display: false } },
                    scales: { y: { beginAtZero: true } }
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
                        document.getElementById('topSources').innerHTML = 
                            stats.top_sources.slice(0, 3).map(ip => `<div style="font-size: 0.9rem; margin: 2px 0;">${ip}</div>`).join('');
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
                    
                    // Update facility chart
                    if (stats.facilities) {
                        const facilityNames = {21: 'Local5', 16: 'Local0', 17: 'Local1', 18: 'Local2', 19: 'Local3', 20: 'Local4', 22: 'Local6', 23: 'Local7'};
                        const facilityData = Object.entries(stats.facilities).map(([k, v]) => ({
                            label: facilityNames[k] || `Facility ${k}`,
                            value: v
                        }));
                        
                        facilityChart.data = {
                            labels: facilityData.map(d => d.label),
                            datasets: [{
                                data: facilityData.map(d => d.value),
                                backgroundColor: '#667eea'
                            }]
                        };
                        facilityChart.update();
                    }
                }
            } catch (error) {
                console.error('Error loading stats:', error);
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

        // Handle Enter key in search
        document.getElementById('searchInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter') searchLogs();
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

@app.get("/api/search")
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
    print("🚀 Starting Syslog ChromaDB FastAPI")
    print(f"📊 Database: {api.db_path}")
    print(f"📄 Documents: {api.collection.count():,}")
    print(f"🔥 Model: {api.model_name}")
    print("🌐 Dashboard: http://localhost:8000")
    print("📋 API Docs: http://localhost:8000/api/docs")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
