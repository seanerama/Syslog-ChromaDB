#!/usr/bin/env python3
"""
Syslog to ChromaDB Pipeline with GPU Embeddings
Real-time ingestion pipeline using HuggingFace transformers and ChromaDB
"""

import socket
import time
import threading
import queue
import json
import hashlib
from datetime import datetime, timezone
from collections import defaultdict, deque
import signal
import sys
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import uuid

# External dependencies (install with: pip install torch transformers chromadb sentence-transformers)
import torch
from transformers import AutoTokenizer, AutoModel
import chromadb
from chromadb.config import Settings
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('syslog_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SyslogMessage:
    """Structured syslog message"""
    timestamp: datetime
    facility: Optional[int]
    severity: Optional[int]
    source_ip: str
    raw_message: str
    parsed_content: str
    message_id: str
    size_bytes: int

class EmbeddingModel:
    """GPU-accelerated embedding model using HuggingFace"""
    
    def __init__(self, model_name: str = "mixedbread-ai/mxbai-embed-large-v1"):
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Handle corporate SSL issues
        self._setup_ssl_bypass()
        
        # Load model and tokenizer
        logger.info(f"Loading model: {model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            logger.info("Falling back to sentence-transformers/all-MiniLM-L6-v2")
            # Fallback to smaller model
            fallback_model = "sentence-transformers/all-MiniLM-L6-v2"
            self.tokenizer = AutoTokenizer.from_pretrained(fallback_model, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(fallback_model, trust_remote_code=True)
            self.model_name = fallback_model
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded successfully on {self.device}")
    
    def _setup_ssl_bypass(self):
        """Setup SSL bypass for corporate networks"""
        import os
        import ssl
        import urllib3
        import requests
        
        # Set cache locations to avoid space issues
        os.environ['HF_HOME'] = '/var/cache/huggingface'
        os.environ['TRANSFORMERS_CACHE'] = '/var/cache/huggingface'
        
        # Disable SSL warnings
        urllib3.disable_warnings()
        
        # Set environment variables
        os.environ['CURL_CA_BUNDLE'] = ''
        os.environ['REQUESTS_CA_BUNDLE'] = ''
        os.environ['SSL_CERT_FILE'] = ''
        
        # Disable SSL verification globally
        ssl._create_default_https_context = ssl._create_unverified_context
        
        # Monkey patch requests
        original_request = requests.adapters.HTTPAdapter.send
        def new_request(self, request, *args, **kwargs):
            kwargs['verify'] = False
            return original_request(self, request, *args, **kwargs)
        requests.adapters.HTTPAdapter.send = new_request
        
    def encode_batch(self, texts: List[str], batch_size: int = 64) -> np.ndarray:
        """Encode a batch of texts to embeddings with optimized batch processing"""
        embeddings = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize with optimized settings
                inputs = self.tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=256,  # Reduced from 512 for speed
                    return_tensors="pt"
                ).to(self.device)
                
                # Get embeddings
                outputs = self.model(**inputs)
                
                # Mean pooling
                attention_mask = inputs['attention_mask']
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                batch_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                
                # Normalize embeddings
                batch_embeddings = torch.nn.functional.normalize(batch_embeddings, p=2, dim=1)
                
                embeddings.append(batch_embeddings.cpu().numpy())
        
        return np.vstack(embeddings)

class ChromaDBManager:
    """ChromaDB client and collection management"""
    
    def __init__(self, db_path: str = "./chromadb", collection_name: str = "syslog_messages", max_size_gb: float = 10.0):
        self.db_path = db_path
        self.collection_name = collection_name
        self.max_size_gb = max_size_gb
        self.max_size_bytes = max_size_gb * 1024 * 1024 * 1024  # Convert to bytes
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(path=db_path)
        
        # Create or get collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Syslog messages with embeddings"}
        )
        
        logger.info(f"ChromaDB initialized at {db_path}, collection: {collection_name}")
        logger.info(f"Current collection size: {self.collection.count()}")
        logger.info(f"Max database size: {max_size_gb}GB")
        
        # Check initial size
        self._check_and_cleanup()
    
    def _get_database_size(self) -> int:
        """Get current database size in bytes"""
        try:
            import os
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(self.db_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    if os.path.exists(filepath):
                        total_size += os.path.getsize(filepath)
            return total_size
        except Exception as e:
            logger.error(f"Error calculating database size: {e}")
            return 0
    
    def _get_oldest_records(self, count: int) -> List[str]:
        """Get IDs of oldest records based on timestamp"""
        try:
            # Get all records with metadata to find oldest
            results = self.collection.query(
                query_texts=[""],  # Dummy query to get all
                n_results=self.collection.count(),
                include=['metadatas', 'ids']
            )
            
            if not results['metadatas'] or not results['metadatas'][0]:
                return []
            
            # Create list of (timestamp, id) pairs
            records = []
            for meta, record_id in zip(results['metadatas'][0], results['ids'][0]):
                timestamp = meta.get('timestamp')
                if timestamp:
                    records.append((timestamp, record_id))
            
            # Sort by timestamp (oldest first) and return IDs
            records.sort(key=lambda x: x[0])
            return [record_id for _, record_id in records[:count]]
            
        except Exception as e:
            logger.error(f"Error finding oldest records: {e}")
            return []
    
    def _cleanup_old_records(self, target_size_bytes: int):
        """Remove oldest records to achieve target size"""
        try:
            current_size = self._get_database_size()
            if current_size <= target_size_bytes:
                return
            
            # Calculate how many records to remove (estimate)
            current_count = self.collection.count()
            if current_count == 0:
                return
            
            # Estimate records to remove (with 20% buffer)
            size_to_remove = current_size - target_size_bytes
            avg_record_size = current_size / current_count
            records_to_remove = int((size_to_remove / avg_record_size) * 1.2)
            
            # Remove in batches to avoid memory issues
            batch_size = min(1000, records_to_remove)
            total_removed = 0
            
            while total_removed < records_to_remove and self.collection.count() > 0:
                oldest_ids = self._get_oldest_records(batch_size)
                if not oldest_ids:
                    break
                
                # Delete the batch
                self.collection.delete(ids=oldest_ids)
                total_removed += len(oldest_ids)
                
                logger.info(f"Removed {len(oldest_ids)} old records (total removed: {total_removed})")
                
                # Check if we've freed enough space
                current_size = self._get_database_size()
                if current_size <= target_size_bytes:
                    break
            
            final_size = self._get_database_size()
            logger.info(f"Cleanup complete: {final_size / (1024**3):.2f}GB remaining")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def _check_and_cleanup(self):
        """Check database size and cleanup if needed"""
        try:
            current_size = self._get_database_size()
            current_size_gb = current_size / (1024**3)
            
            if current_size >= self.max_size_bytes:
                logger.warning(f"Database size ({current_size_gb:.2f}GB) exceeds limit ({self.max_size_gb}GB)")
                logger.info("Starting automatic cleanup of oldest records...")
                
                # Target 80% of max size to avoid frequent cleanups
                target_size = int(self.max_size_bytes * 0.8)
                self._cleanup_old_records(target_size)
            else:
                logger.info(f"Database size: {current_size_gb:.2f}GB / {self.max_size_gb}GB")
        
        except Exception as e:
            logger.error(f"Error in size check: {e}")
    
    def add_messages(self, messages: List[SyslogMessage], embeddings: np.ndarray):
        """Add messages with embeddings to ChromaDB"""
        try:
            # Prepare data for ChromaDB
            ids = [msg.message_id for msg in messages]
            documents = [msg.parsed_content for msg in messages]
            metadatas = []
            
            for msg in messages:
                metadata = {
                    "timestamp": msg.timestamp.isoformat(),
                    "source_ip": msg.source_ip,
                    "facility": msg.facility,
                    "severity": msg.severity,
                    "size_bytes": msg.size_bytes,
                    "raw_message": msg.raw_message[:500]  # Truncate for metadata
                }
                metadatas.append(metadata)
            
            # Add to collection
            self.collection.add(
                embeddings=embeddings.tolist(),
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            logger.info(f"Added {len(messages)} messages to ChromaDB")
            
            # Check size and cleanup if needed (every 10th batch to avoid overhead)
            import random
            if random.randint(1, 10) == 1:  # 10% chance to check
                self._check_and_cleanup()
            
        except Exception as e:
            logger.error(f"Error adding messages to ChromaDB: {e}")
            raise
    
    def search_similar(self, query: str, n_results: int = 10) -> Dict:
        """Search for similar messages"""
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results
        except Exception as e:
            logger.error(f"Error searching ChromaDB: {e}")
            return {}

class SyslogParser:
    """Parse syslog messages into structured format"""
    
    @staticmethod
    def parse_priority(priority_str: str) -> Tuple[Optional[int], Optional[int]]:
        """Parse syslog priority to extract facility and severity"""
        try:
            priority = int(priority_str)
            facility = priority >> 3
            severity = priority & 0x07
            return facility, severity
        except (ValueError, TypeError):
            return None, None
    
    @staticmethod
    def parse_message(raw_message: str, source_ip: str) -> SyslogMessage:
        """Parse raw syslog message into structured format"""
        timestamp = datetime.now(timezone.utc)
        facility = None
        severity = None
        parsed_content = raw_message
        
        # Parse priority if present
        if raw_message.startswith('<') and '>' in raw_message:
            try:
                priority_end = raw_message.index('>')
                priority_str = raw_message[1:priority_end]
                facility, severity = SyslogParser.parse_priority(priority_str)
                parsed_content = raw_message[priority_end + 1:].strip()
            except (ValueError, IndexError):
                pass
        
        # Generate unique message ID
        message_id = hashlib.md5(
            f"{timestamp.isoformat()}{source_ip}{raw_message}".encode()
        ).hexdigest()
        
        return SyslogMessage(
            timestamp=timestamp,
            facility=facility,
            severity=severity,
            source_ip=source_ip,
            raw_message=raw_message,
            parsed_content=parsed_content,
            message_id=message_id,
            size_bytes=len(raw_message.encode('utf-8'))
        )

class SyslogChromaDBPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, 
                 host: str = "0.0.0.0",
                 port: int = 1514,
                 batch_size: int = 20,
                 batch_timeout: float = 30.0,
                 embedding_model: str = "mixedbread-ai/mxbai-embed-large-v1",
                 db_path: str = "./chromadb"):
        
        self.host = host
        self.port = port
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.running = False
        
        # Message queue and batch management
        self.message_queue = queue.Queue(maxsize=10000)  # Even larger buffer
        self.pending_messages = []
        self.last_batch_time = time.time()
        
        # Statistics
        self.stats = {
            'total_received': 0,
            'total_processed': 0,
            'total_errors': 0,
            'batch_count': 0,
            'start_time': None
        }
        
        # Initialize components
        logger.info("Initializing embedding model...")
        self.embedding_model = EmbeddingModel(embedding_model)
        
        logger.info("Initializing ChromaDB...")
        self.chromadb = ChromaDBManager(db_path, max_size_gb=10.0)
        
        # Network socket
        self.sock = None
        
        logger.info("Pipeline initialization complete")
    
    def start_listening(self):
        """Start UDP listener for syslog messages"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.sock.bind((self.host, self.port))
            self.sock.settimeout(1.0)
            
            logger.info(f"Listening for syslog messages on {self.host}:{self.port}")
            
            # Don't set running here - it's already set in main()
            self.stats['start_time'] = time.time()
            
            while self.running:
                try:
                    data, addr = self.sock.recvfrom(4096)
                    raw_message = data.decode('utf-8', errors='ignore')
                    
                    # Parse message
                    message = SyslogParser.parse_message(raw_message, addr[0])
                    
                    # Add to queue
                    try:
                        self.message_queue.put_nowait(message)
                        self.stats['total_received'] += 1
                    except queue.Full:
                        logger.warning("Message queue full, dropping message")
                        self.stats['total_errors'] += 1
                        
                except socket.timeout:
                    continue
                except Exception as e:
                    if self.running:
                        logger.error(f"Error receiving message: {e}")
                        self.stats['total_errors'] += 1
                        
        except Exception as e:
            logger.error(f"Error starting listener: {e}")
        finally:
            if self.sock:
                self.sock.close()
    
    def process_batches(self):
        """Process message batches - embedding and storage"""
        logger.info("Starting batch processor thread")
        
        while self.running:
            try:
                # Get messages from queue
                current_time = time.time()
                batch_ready = False
                
                # Collect messages for batch
                while len(self.pending_messages) < self.batch_size and self.running:
                    try:
                        timeout = max(0.1, self.batch_timeout - (current_time - self.last_batch_time))
                        message = self.message_queue.get(timeout=timeout)
                        self.pending_messages.append(message)
                        current_time = time.time()  # Update time after getting message
                    except queue.Empty:
                        break
                
                # Check if batch is ready
                if (len(self.pending_messages) >= self.batch_size or 
                    (self.pending_messages and current_time - self.last_batch_time >= self.batch_timeout)):
                    batch_ready = True
                
                if batch_ready and self.pending_messages:
                    logger.info(f"Processing batch of {len(self.pending_messages)} messages")
                    self.process_message_batch(self.pending_messages.copy())
                    self.pending_messages.clear()
                    self.last_batch_time = time.time()
                else:
                    # Small delay if no batch is ready
                    time.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error in batch processor: {e}")
                time.sleep(1)
        
        logger.info("Batch processor thread stopped")
    
    def process_message_batch(self, messages: List[SyslogMessage]):
        """Process a batch of messages - create embeddings and store"""
        try:
            start_time = time.time()
            
            # Extract text content for embedding
            texts = [msg.parsed_content for msg in messages]
            
            # Generate embeddings
            logger.debug(f"Generating embeddings for {len(messages)} messages")
            embeddings = self.embedding_model.encode_batch(texts)
            
            # Store in ChromaDB
            self.chromadb.add_messages(messages, embeddings)
            
            # Update statistics
            self.stats['total_processed'] += len(messages)
            self.stats['batch_count'] += 1
            
            processing_time = time.time() - start_time
            logger.info(f"Processed batch of {len(messages)} messages in {processing_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error processing message batch: {e}")
            self.stats['total_errors'] += len(messages)
    
    def display_stats(self):
        """Display pipeline statistics"""
        while self.running:
            try:
                if self.stats['start_time']:
                    uptime = time.time() - self.stats['start_time']
                    rate = self.stats['total_received'] / uptime if uptime > 0 else 0
                    
                    print('\033[2J\033[H', end='')  # Clear screen
                    print("=== SYSLOG -> CHROMADB PIPELINE ===")
                    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Uptime: {int(uptime//3600):02d}:{int((uptime%3600)//60):02d}:{int(uptime%60):02d}")
                    print()
                    
                    print("PROCESSING METRICS:")
                    print(f"  Messages Received: {self.stats['total_received']:,}")
                    print(f"  Messages Processed: {self.stats['total_processed']:,}")
                    print(f"  Messages Queued: {self.message_queue.qsize()}")
                    print(f"  Pending in Batch: {len(self.pending_messages)}")
                    print(f"  Batches Completed: {self.stats['batch_count']}")
                    print(f"  Processing Rate: {rate:.1f} msg/sec")
                    print(f"  Errors: {self.stats['total_errors']}")
                    print()
                    
                    print("CHROMADB STATUS:")
                    db_size = self.chromadb._get_database_size() / (1024**3)
                    print(f"  Total Documents: {self.chromadb.collection.count():,}")
                    print(f"  Database Size: {db_size:.2f}GB / {self.chromadb.max_size_gb}GB")
                    print(f"  Collection: {self.chromadb.collection_name}")
                    print(f"  Database Path: {self.chromadb.db_path}")
                    print()
                    
                    print("GPU STATUS:")
                    if torch.cuda.is_available():
                        print(f"  GPU Memory Used: {torch.cuda.memory_allocated()/1024**3:.1f} GB")
                        print(f"  GPU Memory Cached: {torch.cuda.memory_reserved()/1024**3:.1f} GB")
                    else:
                        print("  Using CPU (no CUDA available)")
                    
                    print("\nPress Ctrl+C to stop pipeline")
                
                time.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in stats display: {e}")
                time.sleep(5)
    
    def search_logs(self, query: str, n_results: int = 10) -> Dict:
        """Search the log database"""
        return self.chromadb.search_similar(query, n_results)
    
    def stop(self):
        """Stop the pipeline gracefully"""
        logger.info("Stopping pipeline...")
        self.running = False
        
        # Process any remaining messages
        if self.pending_messages:
            logger.info(f"Processing final batch of {len(self.pending_messages)} messages")
            self.process_message_batch(self.pending_messages)
        
        # Process any remaining queue items
        remaining_messages = []
        try:
            while True:
                message = self.message_queue.get_nowait()
                remaining_messages.append(message)
        except queue.Empty:
            pass
        
        if remaining_messages:
            logger.info(f"Processing {len(remaining_messages)} remaining messages")
            # Process in batches
            for i in range(0, len(remaining_messages), self.batch_size):
                batch = remaining_messages[i:i + self.batch_size]
                self.process_message_batch(batch)
        
        logger.info("Pipeline stopped gracefully")

def signal_handler(signum, frame):
    """Handle Ctrl+C gracefully"""
    print("\n\nShutting down pipeline...")
    pipeline.stop()
    sys.exit(0)

def main():
    global pipeline
    
    # Set up environment for corporate network and space management
    import os
    import ssl
    import urllib3
    import requests
    
    # Set cache to /var where there's space
    os.environ['HF_HOME'] = '/var/cache/huggingface'
    os.environ['TRANSFORMERS_CACHE'] = '/var/cache/huggingface'
    
    # SSL bypass for corporate network
    ssl._create_default_https_context = ssl._create_unverified_context
    urllib3.disable_warnings()
    original_send = requests.adapters.HTTPAdapter.send
    def bypass_ssl_send(self, request, *args, **kwargs):
        kwargs['verify'] = False
        return original_send(self, request, *args, **kwargs)
    requests.adapters.HTTPAdapter.send = bypass_ssl_send
    
    # Configuration optimized for high-throughput processing
    config = {
        'host': '0.0.0.0',
        'port': 1514,
        'batch_size': 200,  # Larger batches for better GPU efficiency
        'batch_timeout': 2.0,  # Much faster processing
        'embedding_model': 'mixedbread-ai/mxbai-embed-large-v1',  # Use the large, powerful model
        'db_path': '/var/syslog_chromadb'  # Use /var where there's 192G available
    }
    
    logger.info("Starting Syslog -> ChromaDB Pipeline")
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")
    
    # Initialize pipeline
    pipeline = SyslogChromaDBPipeline(**config)
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set running flag BEFORE starting threads
    pipeline.running = True
    
    # Start threads
    listener_thread = threading.Thread(target=pipeline.start_listening, daemon=True)
    processor_thread = threading.Thread(target=pipeline.process_batches, daemon=True)
    stats_thread = threading.Thread(target=pipeline.display_stats, daemon=True)
    
    listener_thread.start()
    processor_thread.start()
    stats_thread.start()
    
    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pipeline.stop()

if __name__ == "__main__":
    main()
