import os
import json
import re
import sqlite3
import time
import logging
import asyncio
from typing import Dict, Any, TypedDict, List, Optional, Tuple, Union
from typing import Set
from functools import lru_cache
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from pathlib import Path
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
import aiohttp
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('agent.log', mode='a')
    ]
)
logger = logging.getLogger(__name__)

# TTL Cache implementation
class TTLCache:
    """Thread-safe TTL cache for expensive operations."""
    
    def __init__(self, ttl_seconds: int = 300):  # 5 minute default TTL
        self.cache: Dict[str, Tuple[Any, float]] = {}
        self.ttl = ttl_seconds
        self.lock = Lock()
    
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.cache:
                value, timestamp = self.cache[key]
                if time.time() - timestamp < self.ttl:
                    logger.debug(f"Cache hit for key: {key}")
                    return value
                else:
                    logger.debug(f"Cache expired for key: {key}")
                    del self.cache[key]
            logger.debug(f"Cache miss for key: {key}")
            return None
    
    def set(self, key: str, value: Any) -> None:
        with self.lock:
            self.cache[key] = (value, time.time())
            logger.debug(f"Cache set for key: {key}")
    
    def clear(self) -> None:
        with self.lock:
            self.cache.clear()
            logger.info("Cache cleared")
    
    def size(self) -> int:
        with self.lock:
            return len(self.cache)

# Circuit Breaker Pattern for API resilience
class CircuitBreaker:
    """Circuit breaker pattern to handle API failures gracefully."""
    
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, recovery_timeout: int = 10):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = Lock()
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
            
        try:
            result = func(*args, **kwargs)
            with self.lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED state")
            return result
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    self.last_failure_time = time.time()
                    logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
                else:
                    logger.warning(f"Circuit breaker failure {self.failure_count}/{self.failure_threshold}")
            raise e
    
    async def async_call(self, func, *args, **kwargs):
        """Async version of circuit breaker call."""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN - service unavailable")
        
        try:
            result = await func(*args, **kwargs)
            with self.lock:
                if self.state == "HALF_OPEN":
                    self.state = "CLOSED"
                    self.failure_count = 0
                    logger.info("Circuit breaker reset to CLOSED state")
            return result
        except Exception as e:
            with self.lock:
                self.failure_count += 1
                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                    self.last_failure_time = time.time()
                    logger.warning(f"Circuit breaker OPENED after {self.failure_count} failures")
                else:
                    logger.warning(f"Circuit breaker failure {self.failure_count}/{self.failure_threshold}")
            raise e

# Connection Pool for SQLite
class SQLiteConnectionPool:
    """Simple SQLite connection pool for better resource management."""
    
    def __init__(self, db_path: str, pool_size: int = 3):
        self.db_path = db_path
        self.pool_size = pool_size
        self.connections: List[sqlite3.Connection] = []
        self.lock = Lock()
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize the connection pool."""
        try:
            for _ in range(self.pool_size):
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                self.connections.append(conn)
            logger.info(f"Initialized SQLite connection pool with {self.pool_size} connections")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            raise
    
    def get_connection(self) -> sqlite3.Connection:
        """Get a connection from the pool."""
        with self.lock:
            if self.connections:
                return self.connections.pop()
            else:
                # If pool is empty, create a new connection
                logger.warning("Connection pool exhausted, creating new connection")
                conn = sqlite3.connect(self.db_path, check_same_thread=False)
                conn.row_factory = sqlite3.Row
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("PRAGMA synchronous=NORMAL;")
                return conn
    
    def return_connection(self, conn: sqlite3.Connection):
        """Return a connection to the pool."""
        with self.lock:
            if len(self.connections) < self.pool_size:
                self.connections.append(conn)
            else:
                # Pool is full, close the connection
                conn.close()
    
    def close_all(self):
        """Close all connections in the pool."""
        with self.lock:
            for conn in self.connections:
                conn.close()
            self.connections.clear()
            logger.info("All connections in pool closed")

# Global caches, connection pool, and circuit breakers
equipment_data_cache = TTLCache(ttl_seconds=600)  # 10 minutes for equipment data
search_results_cache = TTLCache(ttl_seconds=300)  # 5 minutes for search results
connection_pool: Optional[SQLiteConnectionPool] = None

# Circuit breakers for external services
search_circuit_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
openai_circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

# Define a simple tool for demonstration
@tool
def get_current_time() -> str:
    """Get the current time and date."""
    from datetime import datetime
    return f"Current time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

def _calculate_search_result_quality_score(result: Dict[str, Any], query: str, answer_box_content: str = "") -> float:
    """
    Calculate a quality score for a search result based on multiple factors.
    
    Args:
        result: Search result dictionary from Serper API
        query: Original search query
        answer_box_content: Content from answer box if available
        
    Returns:
        Quality score (0-100, higher is better)
    """
    score = 0.0
    
    title = result.get('title', '').lower()
    snippet = result.get('snippet', '').lower()
    link = result.get('link', '').lower()
    query_lower = query.lower()
    
    # Extract key terms from query (manufacturer, model, etc.)
    query_terms = set(re.findall(r'\b\w{2,}\b', query_lower))
    content_text = f"{title} {snippet}".lower()
    
    # 1. Query relevance (30 points max)
    # Count exact query term matches
    title_matches = sum(1 for term in query_terms if term in title)
    snippet_matches = sum(1 for term in query_terms if term in snippet)
    
    relevance_score = min(30, (title_matches * 5) + (snippet_matches * 3))
    score += relevance_score
    
    # 2. Domain authority and trustworthiness (25 points max)
    trusted_domains = {
        # Manufacturer websites (highest trust)
        'keysight.com': 25, 'fluke.com': 25, 'tektronix.com': 25, 'rohde-schwarz.com': 25,
        'bkprecision.com': 25, 'agilent.com': 25, 'ni.com': 25, 'rigol.com': 25,
        'siglent.com': 25, 'keithley.com': 25, 'hioki.com': 25, 'yokogawa.com': 25,
        
        # Technical documentation and spec sites (high trust)
        'datasheetcatalog.com': 20, 'alldatasheet.com': 20, 'datasheet4u.com': 20,
        'electronic-engineering.ch': 20, 'testequipmentdepot.com': 20,
        
        # Industry publications and reviews (good trust)
        'edn.com': 18, 'eetimes.com': 18, 'electronicdesign.com': 18,
        'microwaves101.com': 18, 'rfcafe.com': 18,
        
        # E-commerce and distributors (moderate trust for specs)
        'digikey.com': 15, 'mouser.com': 15, 'newark.com': 15, 'rs-online.com': 15,
        'grainger.com': 15, 'amazon.com': 10, 'ebay.com': 8,
        
        # General tech sites (moderate trust)
        'wikipedia.org': 12, 'electronics-tutorials.ws': 12,
        
        # Forums and Q&A (lower trust but can be valuable)
        'stackoverflow.com': 8, 'reddit.com': 6, 'eevblog.com': 10,
    }
    
    domain_score = 0
    for domain, points in trusted_domains.items():
        if domain in link:
            domain_score = points
            break
    
    # Bonus for HTTPS
    if link.startswith('https://'):
        domain_score += 2
    
    score += min(25, domain_score)
    
    # 3. Content quality indicators (20 points max)
    quality_indicators = {
        # Technical specification indicators
        'specifications': 8, 'datasheet': 8, 'manual': 6, 'spec sheet': 8,
        'technical data': 6, 'user guide': 5, 'operation manual': 6,
        
        # Measurement capability indicators
        'accuracy': 4, 'resolution': 4, 'range': 3, 'measurement': 3,
        'calibration': 5, 'certified': 4, 'accredited': 4,
        
        # Equipment type indicators
        'multimeter': 3, 'oscilloscope': 3, 'spectrum analyzer': 3,
        'signal generator': 3, 'power supply': 3, 'calibrator': 3,
        
        # Electrical measurement terms
        'voltage': 2, 'current': 2, 'resistance': 2, 'frequency': 2,
        'true rms': 4, 'handheld': 2, 'benchtop': 2,
    }
    
    content_quality_score = 0
    for indicator, points in quality_indicators.items():
        if indicator in content_text:
            content_quality_score += points
    
    score += min(20, content_quality_score)
    
    # 4. Title quality (15 points max)
    title_quality_score = 0
    
    # Prefer titles with model numbers
    if re.search(r'\b\d{3,4}[a-z]?\b', title):
        title_quality_score += 5
    
    # Prefer manufacturer names in title
    manufacturers = ['keysight', 'fluke', 'tektronix', 'rohde', 'schwarz', 'agilent', 
                     'ni', 'rigol', 'siglent', 'keithley', 'hioki', 'yokogawa', 'bk precision']
    for manu in manufacturers:
        if manu in title:
            title_quality_score += 3
            break
    
    # Prefer official product pages
    if any(term in title for term in ['datasheet', 'specifications', 'manual', 'overview']):
        title_quality_score += 4
    
    # Avoid generic or commercial titles
    if any(term in title for term in ['buy', 'sale', 'price', 'cheap', 'discount', 'review']):
        title_quality_score -= 2
    
    score += min(15, max(0, title_quality_score))
    
    # 5. Content freshness and uniqueness (10 points max)
    freshness_score = 0
    
    # Look for recent dates in snippet
    if re.search(r'202[0-9]', content_text):
        freshness_score += 3
    elif re.search(r'201[5-9]', content_text):
        freshness_score += 1
    
    # Avoid duplicate content (check against answer box)
    if answer_box_content and len(answer_box_content) > 20:
        # Calculate text similarity to answer box
        answer_terms = set(re.findall(r'\b\w{3,}\b', answer_box_content.lower()))
        content_terms = set(re.findall(r'\b\w{3,}\b', content_text))
        
        if answer_terms and content_terms:
            overlap = len(answer_terms.intersection(content_terms))
            total = len(answer_terms.union(content_terms))
            similarity = overlap / total if total > 0 else 0
            
            # Penalize very similar content (likely duplicate)
            if similarity > 0.7:
                freshness_score -= 3
            elif similarity < 0.3:
                freshness_score += 2  # Reward unique content
    
    score += max(0, freshness_score)
    
    # 6. Query-specific bonus scoring (5 points max)
    query_bonus = 0
    
    # Bonus for specification/datasheet queries
    if any(term in query_lower for term in ['specification', 'datasheet', 'manual', 'spec']):
        if any(term in content_text for term in ['datasheet', 'specifications', 'manual', 'technical data']):
            query_bonus += 3
    
    # Bonus for calibration-related queries
    if any(term in query_lower for term in ['calibration', 'accredited', 'certified']):
        if any(term in content_text for term in ['calibration', 'accredited', 'certified', 'traceable']):
            query_bonus += 2
    
    # Bonus for measurement function queries
    if any(term in query_lower for term in ['voltage', 'current', 'resistance', 'multimeter']):
        if any(term in content_text for term in ['measurement', 'voltage', 'current', 'resistance', 'accuracy']):
            query_bonus += 2
    
    score += query_bonus
    
    # Ensure score is within bounds
    return max(0.0, min(100.0, score))


# Async Google search function for internal use
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
async def _async_google_search(query: str) -> str:
    """
    Async Google search using Serper API with retry logic.
    """
    # Check cache first
    cache_key = f"search:{query.lower().strip()}"
    cached_result = search_results_cache.get(cache_key)
    if cached_result:
        logger.info(f"Returning cached search result for: {query}")
        return cached_result

    logger.info(f"Performing new async search for: {query}")
    
    # Get Serper API key from environment
    serper_api_key = os.getenv("SERPER_API_KEY")
    if not serper_api_key:
        error_msg = "Error: SERPER_API_KEY not found in environment variables."
        logger.error(error_msg)
        return error_msg

    url = "https://google.serper.dev/search"
    payload = {
        "q": query,
        "num": 5  # Return top 5 results
    }
    
    headers = {
        'X-API-KEY': serper_api_key,
        'Content-Type': 'application/json'
    }

    start_time = time.time()
    
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
        async with session.post(url, json=payload, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()
            
            execution_time = time.time() - start_time
            logger.info(f"Async search API call completed in {execution_time:.2f}s")
            
            # Format and rank the search results
            results = []
            
            # Add answer box if available (highest priority)
            answer_box_content = ""
            if 'answerBox' in data:
                answer = data['answerBox']
                if 'answer' in answer:
                    answer_box_content = answer['answer']
                    results.append(f"💡 Quick Answer: {answer_box_content}\n")
                elif 'snippet' in answer:
                    answer_box_content = answer['snippet']
                    results.append(f"💡 Quick Answer: {answer_box_content}\n")
            
            # Rank and add organic results
            if 'organic' in data:
                # Score and rank results for quality
                scored_results = []
                for result in data['organic'][:10]:  # Consider more results for ranking
                    score = _calculate_search_result_quality_score(result, query, answer_box_content)
                    scored_results.append((score, result))
                
                # Sort by score (highest first) and take top 5
                scored_results.sort(key=lambda x: x[0], reverse=True)
                top_results = scored_results[:5]
                
                results.append("🔍 Search Results:")
                for i, (score, result) in enumerate(top_results, 1):
                    title = result.get('title', 'No title')
                    snippet = result.get('snippet', 'No description')
                    link = result.get('link', '')
                    
                    # Add quality indicator for high-scoring results
                    quality_indicator = ""
                    if score >= 80:
                        quality_indicator = " ⭐"
                    elif score >= 60:
                        quality_indicator = " ✓"
                    
                    results.append(f"\n{i}. **{title}**{quality_indicator}")
                    results.append(f"   {snippet}")
                    if link:
                        results.append(f"   🔗 {link}")
                    
                    # Optional: Add debug info for development (remove for production)
                    # if logger.level <= logging.DEBUG:
                    #     results.append(f"   [Quality Score: {score:.1f}]")
            
            if not results:
                result_text = f"No search results found for: {query}"
            else:
                result_text = "\n".join(results)
            
            # Cache the result
            search_results_cache.set(cache_key, result_text)
            logger.info(f"Cached search result for: {query}")
            
            return result_text

# Sync wrapper for backward compatibility
# Define Google search tool using Serper API with caching
@tool  
def google_search(query: str) -> str:
    """
    Search Google for information about a query using Serper API.
    Uses caching, circuit breaker, and async execution for better performance.
    
    Args:
        query: The search query string
        
    Returns:
        Search results as a formatted string
    """
    try:
        # Use circuit breaker to protect against API failures
        def run_async_search():
            # Run async search in a new event loop if none exists
            try:
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # Create a new event loop in a thread if current loop is running
                    with ThreadPoolExecutor() as executor:
                        future = executor.submit(asyncio.run, _async_google_search(query))
                        return future.result(timeout=60)
                else:
                    return loop.run_until_complete(_async_google_search(query))
            except RuntimeError:
                # No event loop exists, create one
                return asyncio.run(_async_google_search(query))
        
        return search_circuit_breaker.call(run_async_search)
        
    except Exception as e:
        error_msg = f"Search failed for '{query}': {str(e)}"
        logger.error(error_msg)
        return error_msg

############################
# Equipment capability tools
############################

# Internal caches for equipment data (DB-backed)
_DB_PATH = os.path.join(os.path.dirname(__file__), "equipment.db")
_ALIAS_TO_MANUFACTURER: Dict[str, str] = {}
_MANUFACTURER_TO_MODELS: Dict[str, List[str]] = {}
_MODEL_TO_RECORDS: Dict[str, List[Dict[str, Any]]] = {}
_ALIAS_MAX_WORDS: int = 1
_LAB_FUNCTION_AREAS: Set[str] = set()


def _normalize_text(value: str) -> str:
    return value.strip().lower()


def _normalize_for_alias(value: str) -> str:
    # Lowercase, replace non-alphanumeric with spaces, collapse multiple spaces
    lowered = value.lower()
    cleaned = re.sub(r"[^a-z0-9]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _collapse_alnum(value: str) -> str:
    """Remove all non-alphanumeric characters and uppercase for model matching."""
    if not value:
        return ""
    return re.sub(r"[^A-Za-z0-9]+", "", str(value)).upper()


def _tokenize_simple(text: str) -> List[str]:
    if not text:
        return []
    return re.findall(r"[A-Za-z0-9]{2,}", text.lower())


def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / max(1, union)


def _extract_description_from_search(search_text: str) -> str:
    """Heuristically extract a short description/snippet from the formatted search output."""
    if not search_text:
        return ""
    # Prefer quick answer
    m = re.search(r"Quick Answer:\s*(.+)", search_text)
    if m:
        return m.group(1).strip()
    # Look for the first indented snippet line (added as '   <snippet>')
    for line in search_text.splitlines():
        if line.strip().startswith("🔗"):
            continue
        if line.startswith("   ") and len(line.strip()) > 20:
            return line.strip()
    # Fallback: first non-empty line with decent length
    for line in search_text.splitlines():
        t = line.strip()
        if len(t) > 30 and not t.startswith("🔍") and not t.startswith("-"):
            return t
    return ""


def _summarize_measurement_functions(text: str, max_labels: int = 8) -> str:
    """Return one sentence summarizing likely measurement functions (conservative).

    Uses ordered, precompiled regex classes to avoid false positives and
    capture a broad set of metrology functions.
    """
    if not text:
        return "Measurement functions unclear from public specs."

    t = text[:800]  # avoid footer/nav noise

    patterns: Dict[str, List[str]] = {
        # Core electrical (enhanced for multimeters)
        "voltage": [
            r"\bvolt(age)?\b", r"\bvdc\b", r"\bvac\b", r"\b[mk]?v\b", r"\btrue\s*rms\b",
            r"\bdc\s*volt(age)?\b", r"\bac\s*volt(age)?\b", r"\b\d+\s*v\b",
        ],
        "current": [
            r"\bcurrent\b", r"\bammeter\b", r"\bclamp\s*meter\b", r"\b[µmu]?a\b", r"\bma\b", r"\ba(?!\w)",
            r"\bdc\s*current\b", r"\bac\s*current\b", r"\b\d+\s*a\b", r"\bamp(ere)?s?\b",
        ],
        "resistance": [
            r"\bresistance\b", r"\bohm[s]?\b", r"Ω", r"\b(4|four)[-\s]?wire\b", r"\bkelvin\s*connection\b",
            r"\bresist(or)?\b", r"\b\d+\s*(ω|ohm)\b", r"\bmω\b", r"\bkω\b",
        ],
        "continuity": [r"\bcontinuity\b", r"\bbeep(er)?\b", r"\bbuzzer\b", r"\bcont\b"],
        "diode": [r"\bdiode\s*test\b", r"\bforward\s*voltage\b", r"\bvf\b", r"\bdiode\b"],
        "multimeter": [r"\bmultimeter\b", r"\bdmm\b", r"\bdigital\s*multimeter\b", r"\bhandheld\s*multimeter\b"],
        "capacitance": [
            r"\bcapacitance\b", r"\bcapacitor\b",
            r"\b\d+(\.\d+)?\s*(pF|nF|uF|µF|mF|F)\b",
            r"\bpf\b|\bnf\b|\buf\b|\bµf\b|\bmf\b",
        ],
        "inductance": [
            r"\binductance\b", r"\blcr\b", r"\b\d+(\.\d+)?\s*([µmu]H|mH|H)\b",
        ],
        "conductance": [r"\bsiemens\b", r"\bmho[s]?\b", r"\bconductance\b"],
        "frequency": [r"\bfrequency\b", r"\b([gmk]?hz)\b"],
        "time_interval": [
            r"\btime\s*interval\b", r"\bperiod\b", r"\bpulse\s*width\b", r"\bduty\s*cycle\b",
            r"\btotalize(r)?\b", r"\bevent\s*count(er)?\b",
        ],
        "temperature": [r"\btemperature\b", r"\bthermocouple\b", r"\brtd\b", r"\bpt1000?\b", r"\bthermistor\b", r"°c|°f"],

        # Oscilloscope / power analysis
        "oscilloscope": [r"\boscilloscope\b", r"\bscope\b", r"\bbandwidth\b", r"\bsample\s*rate\b", r"\bwaveform\b", r"\btrigger\b", r"\bchannels?\b"],
        "power_analysis": [r"\bpower\s*analy(ser|zer)\b", r"\bwatts?\b", r"\bva\b", r"\bvar\b", r"\bpower\s*factor\b", r"\bharmonics\b", r"\bthd\b"],

        # RF & network
        "rf_spectrum": [r"\bspectrum\s*analy(ser|zer)\b", r"\bsignal\s*analy(ser|zer)\b", r"\bvector\s*signal\s*analy(ser|zer)\b", r"\bchannel\s*power\b", r"\bacpr\b", r"\boccupied\s*bandwidth\b", r"\bevm\b", r"\bspectrum\s*mask\b"],
        "network_analysis": [r"\bnetwork\s*analy(ser|zer)\b", r"\bvna\b", r"\bs-?parameters?\b", r"\breturn\s*loss\b", r"\bvswr\b", r"\bsmith\s*chart\b"],
        "rf_power": [r"\brf\s*power\b", r"\bpower\s*meter\b"],

        # Safety test
        "insulation_resistance": [r"\binsulation\s*resistance\b", r"\bmegohm(meter)?\b", r"\bgΩ\b", r"\bgohm[s]?\b"],
        "hipot": [r"\bhi[-\s]?pot\b", r"\bdielectric\s*withstand\b", r"\bdwv\b", r"\bwithstand\s*test\b"],
        "ground_bond": [r"\bground\s*bond\b", r"\bearth\s*bond\b", r"\bprotective\s*earth\b", r"\bpe\s*resistance\b"],

        # Pressure / flow / environment
        "pressure": [
            r"\bpressure\b",
            r"\b\d+(\.\d+)?\s*(psi|in\s*h2o|in\s*h₂o|inh2o|in\s*h[gG]|inhg|mmhg|torr|mbar|bar|kpa|mpa|pa)\b",
        ],
        "flow": [r"\bflow\s*(meter|rate|controller)\b|\bmfc\b", r"\b\d+(\.\d+)?\s*(slpm|sccm|l/min|lpm|scfh|cfm)\b"],
        "humidity": [r"\brelative\s*humidity\b|\b%\s*rh\b|\bhumidity\b|\bdew\s*point\b|\bwet\s*bulb\b"],

        # Mechanical
        "mass_weight": [r"\bweigh(ing)?\b|\bbalance\b|\bscale\b|\bmass\s*comparator\b", r"\b\d+(\.\d+)?\s*(g|kg|lb|oz)\b"],
        "force": [r"\bforce\s*(gauge|meter)?\b", r"\b\d+(\.\d+)?\s*(n|kn|lbf)\b"],
        "torque": [r"\btorque\b", r"\b\d+(\.\d+)?\s*(n·m|n-m|nm|oz[-\s]?in|lb[-\s]?(in|ft))\b"],
        "rpm_tach": [r"\brpm\b", r"\btach(ometer)?\b", r"\bstroboscope\b"],
        "vibration": [r"\bvibration\b|\baccelerometer\b", r"\b\d+(\.\d+)?\s*(g|m/s\^?2)\b"],

        # Light / sound
        "light_lux": [r"\blux\b|\billuminance\b|\bphotometer\b|\blight\s*meter\b"],
        "sound_level": [r"\bsound\s*level\s*meter\b|\bspl\b|\bdba\b|\bdbc\b|\bdosimeter\b"],
    }

    order: List[str] = [
        "multimeter","voltage","current","resistance","continuity","diode",
        "capacitance","inductance","conductance",
        "frequency","time_interval","temperature",
        "oscilloscope","power_analysis",
        "rf_spectrum","network_analysis","rf_power",
        "insulation_resistance","hipot","ground_bond",
        "pressure","flow","humidity",
        "mass_weight","force","torque","rpm_tach","vibration",
        "light_lux","sound_level",
    ]

    found: List[str] = []
    for label in order:
        for patt in patterns[label]:
            if re.search(patt, t, flags=re.IGNORECASE):
                if label not in found:
                    found.append(label)
                break

    if not found:
        return "Measurement functions unclear from public specs."
    return "Likely measurement functions include: " + ", ".join(found[:max_labels]) + "."


def _preferred_domains_for_manu(manufacturer: str) -> List[str]:
    manu = (manufacturer or "").upper()
    mapping = {
        "KEYSIGHT": ["keysight.com", "agilent.com", "hp.com"],
        "FLUKE": ["fluke.com"],
        "FLUKE CALIBRATION": ["flukecal.com", "fluke.com"],
        "TEKTRONIX": ["tek.com"],
        "ROHDE & SCHWARZ": ["rohde-schwarz.com"],
        "TELEDYNE LECROY": ["teledynelecroy.com", "lecroy.com"],
        "BK PRECISION": ["bkprecision.com"],
        "GW INSTEK": ["gwinstek.com"],
        "RIGOL": ["rigolna.com", "rigol.com"],
        "SIGLENT": ["siglent.com"],
        "YOKOGAWA": ["yokogawa.com"],
        "ANRITSU": ["anritsu.com"],
        "VIAVI SOLUTIONS": ["viavisolutions.com"],
    }
    return mapping.get(manu, [])


def _extract_snippet_near_link(search_text: str, preferred_domains: List[str]) -> Optional[str]:
    if not search_text:
        return None
    lines = search_text.splitlines()
    for i, line in enumerate(lines):
        if "🔗" in line and "http" in line:
            # Extract URL
            m = re.search(r"https?://[^\s]+", line)
            if not m:
                continue
            url = m.group(0)
            if any(dom in url for dom in preferred_domains):
                # The snippet is usually the previous indented line
                # Find previous non-empty line
                for j in range(i - 1, -1, -1):
                    prev = lines[j].strip()
                    if prev and not prev.startswith("🔍") and not prev.startswith("**"):
                        return prev
    return None


def _expand_search_queries(manufacturer: str, model: str, query: str) -> List[str]:
    manu = (manufacturer or "").strip()
    modl = (model or "").strip()
    domains = _preferred_domains_for_manu(manu)

    expanded: List[str] = []
    core = " ".join([p for p in [manu, modl] if p]) or query
    if core:
        expanded.append(core)
        expanded.append(core + " datasheet")
        expanded.append(core + " manual")
        expanded.append(core + " specifications")
        expanded.append(core + " overview")
        expanded.append(core + " True RMS multimeter")
    if modl and domains:
        for d in domains:
            expanded.append(f"site:{d} {modl}")
            expanded.append(f"site:{d} {modl} datasheet")
            expanded.append(f"site:{d} {manu} {modl}")
    # Deduplicate while preserving order
    seen = set()
    uniq: List[str] = []
    for q in expanded:
        if q and q not in seen:
            uniq.append(q)
            seen.add(q)
    return uniq


def _get_db_connection() -> sqlite3.Connection:
    """Get a database connection from the pool."""
    global connection_pool
    if connection_pool is None:
        try:
            connection_pool = SQLiteConnectionPool(_DB_PATH)
            logger.info("Initialized database connection pool")
        except Exception as e:
            logger.error(f"Failed to initialize connection pool: {e}")
            # Fallback to direct connection
            conn = sqlite3.connect(_DB_PATH)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            return conn
    
    return connection_pool.get_connection()

def _return_db_connection(conn: sqlite3.Connection) -> None:
    """Return a database connection to the pool."""
    global connection_pool
    if connection_pool is not None:
        connection_pool.return_connection(conn)
    else:
        conn.close()


def _load_equipment_data() -> None:
    """Load equipment data with intelligent caching."""
    global _ALIAS_TO_MANUFACTURER, _MANUFACTURER_TO_MODELS, _MODEL_TO_RECORDS, _ALIAS_MAX_WORDS
    
    # Check if already loaded in memory
    if _MANUFACTURER_TO_MODELS and _MODEL_TO_RECORDS and _ALIAS_TO_MANUFACTURER:
        logger.debug("Equipment data already loaded in memory")
        return

    # Check cache
    cache_key = "equipment_data"
    cached_data = equipment_data_cache.get(cache_key)
    if cached_data:
        logger.info("Loading equipment data from cache")
        _ALIAS_TO_MANUFACTURER = cached_data["alias_to_manufacturer"]
        _MANUFACTURER_TO_MODELS = cached_data["manufacturer_to_models"]
        _MODEL_TO_RECORDS = cached_data["model_to_records"]
        _ALIAS_MAX_WORDS = cached_data["alias_max_words"]
        return

    logger.info("Loading equipment data from database")
    start_time = time.time()

    alias_map: Dict[str, str] = {}
    manu_to_models: Dict[str, List[str]] = {}
    model_to_records: Dict[str, List[Dict[str, Any]]] = {}
    alias_max_words = 1

    conn = None
    try:
        # Load alias CSVs from builder to enrich manufacturer alias detection
        try:
            from build_equipment_db import load_aliases, normalize_manufacturer
        except Exception as e:
            logger.warning(f"Could not import build_equipment_db helpers: {e}")
            load_aliases = None  # type: ignore
            normalize_manufacturer = lambda s: (s or "").strip().upper()  # type: ignore

        if load_aliases is not None:
            try:
                aliases = load_aliases(Path(_DB_PATH).parent)
                # Map normalized alias (our alias-normalizer) to canonical display name
                for alias_norm, canonical in aliases.manufacturer_alias_to_canonical.items():
                    alias_key = _normalize_for_alias(alias_norm)
                    alias_map[alias_key] = canonical
                logger.info(f"Loaded {len(alias_map)} manufacturer aliases")
            except Exception as e:
                logger.warning(f"Failed to load aliases: {e}")

        conn = _get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT manufacturer, model, description, accredited FROM equipment")
        rows = cur.fetchall()
        
        logger.info(f"Processing {len(rows)} equipment records")

        for r in rows:
            manufacturer = (r["manufacturer"] or "").strip()
            model_num = (r["model"] or "").strip()
            description = r["description"] or ""
            accredited = int(r["accredited"]) if r["accredited"] is not None else 0

            if not manufacturer or not model_num:
                continue

            # Alias map: include manufacturer self
            alias_map.setdefault(_normalize_for_alias(manufacturer), manufacturer)
            alias_max_words = max(alias_max_words, len(_normalize_for_alias(manufacturer).split()))

            # Manufacturer -> models
            manu_to_models.setdefault(manufacturer, [])
            manu_to_models[manufacturer].append(model_num)

            # Model -> records
            key = _normalize_text(model_num)
            model_to_records.setdefault(key, [])
            model_to_records[key].append({
                "manufacturer": manufacturer,
                "model_num": model_num,
                "gage_descr": description,
                "accredited": accredited,
            })

        execution_time = time.time() - start_time
        logger.info(f"Loaded equipment data in {execution_time:.2f}s: {len(manu_to_models)} manufacturers, {len(model_to_records)} models")

        # Cache the data
        cached_data = {
            "alias_to_manufacturer": alias_map,
            "manufacturer_to_models": manu_to_models,
            "model_to_records": model_to_records,
            "alias_max_words": alias_max_words
        }
        equipment_data_cache.set(cache_key, cached_data)

    except Exception as e:
        logger.error(f"Error loading equipment data: {e}")
        # Keep empty maps for graceful degradation
        alias_map = {}
        manu_to_models = {}
        model_to_records = {}
        alias_max_words = 1
    finally:
        if conn:
            _return_db_connection(conn)

    _ALIAS_TO_MANUFACTURER = alias_map
    _MANUFACTURER_TO_MODELS = manu_to_models
    _MODEL_TO_RECORDS = model_to_records
    _ALIAS_MAX_WORDS = alias_max_words


def _invalidate_equipment_cache() -> None:
    """Clear both memory and TTL cache for equipment data."""
    global _ALIAS_TO_MANUFACTURER, _MANUFACTURER_TO_MODELS, _MODEL_TO_RECORDS
    logger.info("Invalidating equipment cache")
    
    # Clear memory cache
    _ALIAS_TO_MANUFACTURER = {}
    _MANUFACTURER_TO_MODELS = {}
    _MODEL_TO_RECORDS = {}
    
    # Clear TTL cache
    equipment_data_cache.clear()


def _find_manufacturer_in_text(text: str) -> Optional[str]:
    _load_equipment_data()
    if not _ALIAS_TO_MANUFACTURER:
        return None

    # Generate n-grams from the text and look up directly in alias map
    norm_text = _normalize_for_alias(text)
    if not norm_text:
        return None
    tokens = norm_text.split()
    max_n = min(_ALIAS_MAX_WORDS, len(tokens))

    # Try longer n-grams first for specificity
    for n in range(max_n, 0, -1):
        for i in range(0, len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i+n])
            if ngram in _ALIAS_TO_MANUFACTURER:
                return _ALIAS_TO_MANUFACTURER[ngram]
    return None


def _find_model_for_manufacturer(text: str, manufacturer: str) -> Optional[str]:
    _load_equipment_data()
    models = _MANUFACTURER_TO_MODELS.get(manufacturer, [])
    
    # If no models found for exact manufacturer name, search for partial matches
    if not models:
        potential_manufacturers = []
        for db_manufacturer in _MANUFACTURER_TO_MODELS.keys():
            if manufacturer.upper() in db_manufacturer.upper():
                potential_manufacturers.append(db_manufacturer)
        
        # Combine models from all potential manufacturers
        for manu in potential_manufacturers:
            models.extend(_MANUFACTURER_TO_MODELS.get(manu, []))
    
    if not models:
        return None
        
    # Prefer longest model strings first; match allowing optional separators
    text_norm = _collapse_alnum(text)
    for model in sorted(models, key=len, reverse=True):
        if not model:
            continue
        model_norm = _collapse_alnum(model)
        if model_norm and model_norm in text_norm:
            return model
    return None


def _find_model_any_manufacturer(text: str) -> Tuple[Optional[str], Optional[str]]:
    """Return (manufacturer, model) if a model appears in text without known manufacturer."""
    _load_equipment_data()
    if not _MODEL_TO_RECORDS:
        return None, None

    # Allow space/dash variants by collapsing input and comparing with collapsed model strings
    text_norm = _collapse_alnum(text)

    # Gather candidate records whose collapsed model starts with the input (strict to avoid false positives)
    candidates: List[Tuple[str, str]] = []  # (manufacturer, model)
    manufacturers: List[str] = []
    for recs in _MODEL_TO_RECORDS.values():
        for rec in recs:
            model = rec.get("model_num") or ""
            if not model:
                continue
            m_norm = _collapse_alnum(model)
            if not m_norm or not text_norm:
                continue
            if m_norm.startswith(text_norm) or m_norm == text_norm:
                candidates.append((rec.get("manufacturer"), model))
                manufacturers.append(rec.get("manufacturer"))

    if not candidates:
        return None, None

    # If only one manufacturer among candidates, return that one
    unique_manu = {m for m in manufacturers if m}
    if len(unique_manu) == 1:
        mfg = next(iter(unique_manu))
        # pick the longest model string for specificity
        best_model = sorted([m for mm, m in candidates if mm == mfg], key=len, reverse=True)[0]
        return mfg, best_model

    # Ambiguous manufacturer; leave manufacturer unresolved and return a representative model (longest)
    best_model = sorted([m for _, m in candidates], key=len, reverse=True)[0]
    return None, best_model
    return None, None


@tool
def parse_equipment_details(user_text: str) -> str:
    """Parse manufacturer and model from a free-text equipment description. Returns a JSON string with keys: manufacturer, model, strategy."""
    # Input validation
    if not user_text or not isinstance(user_text, str):
        logger.warning("Invalid input to parse_equipment_details")
        return json.dumps({"error": "Invalid input: user_text must be a non-empty string"})
    
    if len(user_text.strip()) < 2:
        return json.dumps({"error": "Input text too short to parse"})
    
    try:
        logger.info(f"Parsing equipment details for: {user_text[:50]}...")
        start_time = time.time()
        
        _load_equipment_data()
        manufacturer = _find_manufacturer_in_text(user_text)
        model: Optional[str] = None
        strategy = ""

        if manufacturer:
            model = _find_model_for_manufacturer(user_text, manufacturer)
            strategy = "manufacturer_first"
        if not model:
            # Try model-first discovery
            manufacturer2, model2 = _find_model_any_manufacturer(user_text)
            if model2:
                manufacturer = manufacturer or manufacturer2
                model = model2
                strategy = "model_first"

        # Enhanced heuristic fallback: extract model patterns more aggressively
        if not model:
            text = user_text.strip()
            # Enhanced pattern matching for model numbers
            import re
            # Pattern 1: Letters followed by digits (like "394B", "3458A")
            pattern1 = re.findall(r'\b([A-Za-z]*\d+[A-Za-z]*)\b', text.upper())
            # Pattern 2: Traditional model patterns with optional separators
            pattern2 = re.findall(r"\b([A-Za-z]{2,}[A-Za-z0-9]*[-_]?\d+[A-Za-z0-9]*)\b", text)
            
            # Combine patterns and take the longest match
            all_matches = pattern1 + pattern2
            if all_matches:
                model = max(all_matches, key=len)
                strategy = strategy or "heuristic_enhanced"
                logger.info(f"Enhanced heuristic extracted model: {model}")
                
                # If we have a model but no manufacturer, try to extract manufacturer
                if not manufacturer:
                    # Look for manufacturer before the model
                    model_pos = text.upper().find(model.upper())
                    if model_pos > 0:
                        prefix = text[:model_pos].strip()
                        words = re.findall(r"[A-Za-z][A-Za-z&.]+", prefix)
                        if words:
                            manu_guess = " ".join(words[-3:]).strip()
                            manufacturer = manu_guess
                            logger.info(f"Extracted manufacturer from prefix: {manufacturer}")
                    
        # Legacy heuristic fallback (keep for compatibility)
        if not manufacturer or not model:
            text = user_text.strip()
            # Find token with letters+digits optionally with hyphen/underscore
            m = re.search(r"\b([A-Za-z]{2,}[A-Za-z0-9]*[-_]?\d+[A-Za-z0-9]*)\b", text)
            if m:
                cand_model = m.group(1)
                # Manufacturer as up to 3 words before the model
                prefix = text[: m.start()].strip()
                words = re.findall(r"[A-Za-z][A-Za-z&.]+", prefix)
                if words:
                    manu_guess = " ".join(words[-3:]).strip()
                    if not manufacturer and manu_guess:
                        manufacturer = manu_guess
                if not model:
                    model = cand_model
                strategy = strategy or "heuristic"

        execution_time = time.time() - start_time
        logger.info(f"Parsed equipment in {execution_time:.2f}s: {manufacturer}/{model} via {strategy}")

        result = {
            "manufacturer": manufacturer,
            "model": model,
            "strategy": strategy or "none",
        }
        return json.dumps(result)
    except Exception as e:
        logger.error(f"Error parsing equipment details: {e}")
        return json.dumps({"error": f"Parsing failed: {str(e)}"});


@tool
def check_lab_capability(manufacturer: str, model: str) -> str:
    """Compare a manufacturer and model against Tescom's equipment DB.
    Returns a JSON string with keys:
      - supported: bool (True if any DB match found)
      - accredited_level1: bool | null (True if any matched record has accredited==1; null when not supported)
      - matches: list of {manufacturer, model, description, accredited}
      - resolved: {manufacturer, model_query, ambiguous}
    """
    # Input validation
    if manufacturer and not isinstance(manufacturer, str):
        logger.warning("Invalid manufacturer input to check_lab_capability")
        return json.dumps({"error": "Manufacturer must be a string"})
    
    if model and not isinstance(model, str):
        logger.warning("Invalid model input to check_lab_capability")
        return json.dumps({"error": "Model must be a string"})
    
    try:
        logger.info(f"Checking lab capability for: {manufacturer}/{model}")
        start_time = time.time()
        
        _load_equipment_data()
        if not manufacturer and not model:
            return json.dumps({
                "supported": False,
                "accredited_level1": None,
                "matches": [],
                "resolved": {"manufacturer": None, "model_query": model or "", "ambiguous": False},
                "error": "No manufacturer or model provided."
            })

        # Find canonical manufacturer via alias map if input is an alias
        canonical_manu = _ALIAS_TO_MANUFACTURER.get(_normalize_for_alias(manufacturer), manufacturer)
        
        # Additional step: if canonical_manu doesn't exist in our models cache, 
        # search for manufacturers that contain the canonical manufacturer
        if canonical_manu not in _MANUFACTURER_TO_MODELS:
            # Look for partial matches (e.g., "KEYSIGHT" should match "AGILENT/HP/KEYSIGHT")
            potential_matches = []
            for db_manufacturer in _MANUFACTURER_TO_MODELS.keys():
                if canonical_manu.upper() in db_manufacturer.upper():
                    potential_matches.append(db_manufacturer)
            
            # If we found exactly one match, use it
            if len(potential_matches) == 1:
                logger.info(f"Resolved manufacturer alias '{manufacturer}' -> '{canonical_manu}' -> '{potential_matches[0]}'")
                canonical_manu = potential_matches[0]
            elif len(potential_matches) > 1:
                # Multiple matches - prefer the one with the most words (most specific)
                best_match = max(potential_matches, key=lambda x: len(x.split()))
                logger.info(f"Multiple manufacturer matches for '{canonical_manu}', using most specific: '{best_match}'")
                canonical_manu = best_match

        matches: List[Dict[str, Any]] = []
        if model:
            recs = _MODEL_TO_RECORDS.get(_normalize_text(model), [])
            if canonical_manu:
                matches = [r for r in recs if r.get("manufacturer") == canonical_manu]
            else:
                matches = recs

        # If no exact model match, try partial/normalized match
        if not matches and model:
            text_norm = _collapse_alnum(model)
            if canonical_manu:
                models = _MANUFACTURER_TO_MODELS.get(canonical_manu, [])
                for m in models:
                    if not m:
                        continue
                    m_norm = _collapse_alnum(m)
                    if m_norm and (m_norm.startswith(text_norm) or m_norm == text_norm):
                        matches.extend(_MODEL_TO_RECORDS.get(_normalize_text(m), []))
            else:
                # Search across all manufacturers
                for recs in _MODEL_TO_RECORDS.values():
                    for rec in recs:
                        m = rec.get("model_num") or ""
                        if not m:
                            continue
                        m_norm = _collapse_alnum(m)
                        if m_norm and (m_norm.startswith(text_norm) or m_norm == text_norm):
                            matches.append(rec)

        if not matches:
            ambiguous = False
            
            # Auto-trigger research for unsupported equipment
            if manufacturer and model:
                query = f"{manufacturer} {model}".strip()
            elif manufacturer:
                query = manufacturer
            elif model:
                query = model
            else:
                query = "unknown equipment"
            
            try:
                logger.info(f"Auto-triggering research for unsupported equipment: {query}")
                research = infer_capability_via_research.invoke({"query": query})
                research_obj = json.loads(research)
                
                # Merge research results
                result = {
                    "supported": False,
                    "accredited_level1": None,
                    "matches": [],
                    "resolved": {"manufacturer": canonical_manu if canonical_manu != manufacturer else manufacturer or None, "model_query": model or "", "ambiguous": ambiguous},
                    "note": "No exact or partial match found",
                    "summary": "The equipment is not supported in Tescom's database.",
                    "fallback_research": research_obj,
                    "fallback_helper": {
                        "description": research_obj.get("description", ""),
                        "functions_summary": research_obj.get("functions_summary", ""),
                        "overlaps": research_obj.get("overlaps", []),
                        "likely_supported": research_obj.get("likely_supported", False)
                    }
                }
                return json.dumps(result)
            except Exception as e:
                logger.error(f"Auto-research failed: {e}")
                # Fall back to original response
                pass
            
            return json.dumps({
                "supported": False,
                "accredited_level1": None,
                "matches": [],
                "resolved": {"manufacturer": canonical_manu if canonical_manu != manufacturer else manufacturer or None, "model_query": model or "", "ambiguous": ambiguous},
                "note": "No exact or partial match found",
                "summary": "The equipment is not supported in Tescom's database."
            })

        # Deduplicate and build structured matches
        seen = set()
        match_items = []
        manufacturers_seen = set()
        for rec in matches:
            key = (rec.get("manufacturer"), rec.get("model_num"), rec.get("gage_descr"))
            if key in seen:
                continue
            seen.add(key)
            manufacturers_seen.add(rec.get("manufacturer"))
            match_items.append({
                "manufacturer": rec.get("manufacturer"),
                "model": rec.get("model_num"),
                "description": rec.get("gage_descr"),
                "accredited": int(rec.get("accredited", 0)),
            })

        supported = len(match_items) > 0
        accredited_any = any(int(m["accredited"]) == 1 for m in match_items)
        ambiguous = len(manufacturers_seen) > 1 and not canonical_manu

        # Human-readable summary per requested phrasing
        if supported and accredited_any:
            first = match_items[0]
            summary = f"The equipment {first['manufacturer']} {first['model']} is supported and can be given accredited calibration (Level 1)."
        elif supported and not accredited_any:
            first = match_items[0]
            summary = f"The equipment {first['manufacturer']} {first['model']} is supported, but not at Level 1."
        else:
            summary = "The equipment is not supported in Tescom's database."

        execution_time = time.time() - start_time
        logger.info(f"Capability check completed in {execution_time:.2f}s: {len(match_items)} matches, supported={supported}")

        return json.dumps({
            "supported": bool(supported),
            "accredited_level1": bool(accredited_any) if supported else None,
            "matches": match_items,
            "resolved": {
                "manufacturer": canonical_manu if canonical_manu else None,
                "model_query": model or "",
                "ambiguous": bool(ambiguous),
            },
            "summary": summary,
        })
    except Exception as e:
        logger.error(f"Error checking capability for {manufacturer}/{model}: {e}")
        return json.dumps({"error": f"Error checking capability: {str(e)}"})


############################
# Update agent tools (JSON editor)
############################

def _atomic_write_json(path: str, data: Any) -> None:
    import tempfile
    import shutil
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=directory, encoding="utf-8") as tmp:
        json.dump(data, tmp, ensure_ascii=False, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        temp_path = tmp.name
    shutil.move(temp_path, path)


def _backup_json(path: str) -> str:
    from datetime import datetime
    base_dir = os.path.dirname(path)
    backups_dir = os.path.join(base_dir, "backups")
    os.makedirs(backups_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = os.path.join(backups_dir, f"tescom_equip_list.{ts}.json")
    with open(path, "r", encoding="utf-8") as fsrc, open(backup_path, "w", encoding="utf-8") as fdst:
        fdst.write(fsrc.read())
    return backup_path


def _append_change_log(entry: Dict[str, Any]) -> None:
    from datetime import datetime
    entry = dict(entry)
    entry["timestamp"] = datetime.utcnow().isoformat() + "Z"
    log_path = os.path.join(os.path.dirname(__file__), "capability_changes.log")
    line = json.dumps(entry, ensure_ascii=False)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


@tool
def parse_update_request(request_text: str) -> str:
    """Parse a free-text capability update like 'Keysight 34401A is accredited' or 'Fluke 8508A not accredited'.
    Returns JSON: {manufacturer, model, new_accredited} where new_accredited is 0 or 1.
    """
    try:
        text = request_text.strip()
        # Determine accredited intent
        new_accredited: Optional[int] = None
        if re.search(r"\bnot\s+accredit(ed|able)?\b", text, flags=re.IGNORECASE):
            new_accredited = 0
        elif re.search(r"\b(non[- ]?accredited|unaccredited)\b", text, flags=re.IGNORECASE):
            new_accredited = 0
        elif re.search(r"\b(accredited|accreditable|can\s+be\s+accredited)\b", text, flags=re.IGNORECASE):
            new_accredited = 1
        elif re.search(r"\b(no)\b", text, flags=re.IGNORECASE):
            new_accredited = 0
        elif re.search(r"\b(yes)\b", text, flags=re.IGNORECASE):
            new_accredited = 1

        # Reuse existing parsing for manufacturer/model
        parsed = json.loads(parse_equipment_details.invoke({"user_text": request_text}))
        manufacturer = parsed.get("manufacturer")
        model = parsed.get("model")

        return json.dumps({
            "manufacturer": manufacturer,
            "model": model,
            "new_accredited": new_accredited,
        })
    except Exception as e:
        return json.dumps({"error": str(e)})


@tool
def apply_update(manufacturer: str, model: str, new_accredited: int, user_name: str, password: str) -> str:
    """Apply an accredited flag update in equipment.db after password check.
    Password is 'bigbrain' (case-insensitive). new_accredited must be 0 or 1.
    Returns a concise summary of changes or an error.
    """
    try:
        if not password or password.lower().strip() != "bigbrain":
            return "Error: Invalid password."
        if new_accredited not in (0, 1):
            return "Error: new_accredited must be 0 or 1."
        if not manufacturer or not model:
            return "Error: manufacturer and model are required."

        # Normalize and apply aliases consistent with DB norms
        try:
            from build_equipment_db import load_aliases, apply_manufacturer_alias, apply_model_alias, normalize_manufacturer, normalize_model
        except Exception as e:
            return f"Error: cannot import alias helpers: {e}"

        aliases = load_aliases(Path(_DB_PATH).parent)
        manu_canon = apply_manufacturer_alias(manufacturer, aliases)
        manu_norm = normalize_manufacturer(manu_canon)
        model_norm = normalize_model(model)
        model_norm = apply_model_alias(manu_norm, model_norm, aliases)

        conn = None
        try:
            conn = _get_db_connection()
            cur = conn.cursor()
            cur.execute(
                "UPDATE equipment SET accredited = ? WHERE manufacturer_norm = ? AND model_norm = ?",
                (int(new_accredited), manu_norm, model_norm),
            )
            changed = cur.rowcount
            conn.commit()
            logger.info(f"Database update completed: {changed} rows affected")
        finally:
            if conn:
                _return_db_connection(conn)

        if changed == 0:
            return f"No records found for {manu_canon} {model}. No changes applied."

        _append_change_log({
            "action": "update_accredited",
            "user": user_name or "unknown",
            "manufacturer": manu_canon,
            "model": model,
            "new_accredited": int(new_accredited),
            "rows_updated": int(changed),
        })

        # Invalidate caches so next lookup reflects changes
        _invalidate_equipment_cache()

        return f"Updated accredited={int(new_accredited)} for {int(changed)} record(s) matching {manu_canon} {model}."
    except Exception as e:
        return f"Error applying update: {str(e)}"


# Initialize the LLM with tools
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.2,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=1,
    timeout=30,
    max_tokens=300,
)

# Update agent LLM (higher-accuracy model)
update_llm = ChatOpenAI(
    model="gpt-4.1",
    temperature=0.0,
    api_key=os.getenv("OPENAI_API_KEY"),
    max_retries=1,
    timeout=45,
)

# Bind tools to the orchestrator LLM
llm_with_tools = llm.bind_tools([
    get_current_time,
    google_search,
    parse_equipment_details,
    check_lab_capability,
    # Orchestrator will also be able to trigger parsing updates, but not apply them directly
    parse_update_request,
    apply_update,
])

# Bind tools to the update LLM
update_llm_with_tools = update_llm.bind_tools([
    parse_update_request,
    apply_update,
])

# Define the state structure using TypedDict for better compatibility
class AgentState(TypedDict):
    messages: List[HumanMessage | AIMessage | SystemMessage]

# Add cleanup function for graceful shutdown
def cleanup_resources():
    """Clean up resources on shutdown."""
    global connection_pool
    if connection_pool:
        logger.info("Closing database connection pool")
        connection_pool.close_all()

# Async tool execution helper
async def execute_tool_async(tool_call: Dict[str, Any]) -> str:
    """Execute a single tool call asynchronously."""
    name = tool_call.get("name")
    args = tool_call.get("args", {}) or {}
    
    try:
        if name == "get_current_time":
            return get_current_time.invoke({})
        elif name == "google_search":
            # Use the async version directly for better performance
            query = args.get("query", "")
            try:
                return await search_circuit_breaker.async_call(_async_google_search, query)
            except Exception as e:
                logger.error(f"Async search failed, falling back to sync: {e}")
                return google_search.invoke({"query": query})
        elif name == "parse_equipment_details":
            return parse_equipment_details.invoke({"user_text": args.get("user_text", "")})
        elif name == "check_lab_capability":
            return check_lab_capability.invoke({
                "manufacturer": args.get("manufacturer", ""),
                "model": args.get("model", ""),
            })
        elif name == "parse_update_request":
            return parse_update_request.invoke({
                "request_text": args.get("request_text", "")
            })
        elif name == "apply_update":
            return apply_update.invoke({
                "manufacturer": args.get("manufacturer", ""),
                "model": args.get("model", ""),
                "new_accredited": args.get("new_accredited", 0),
                "user_name": args.get("user_name", ""),
                "password": args.get("password", ""),
            })
        else:
            return f"Tool '{name}' not recognized."
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}")
        return f"Error executing {name}: {str(e)}"

# Concurrent tool execution function
async def execute_tools_concurrently(tool_calls: List[Dict[str, Any]]) -> List[Tuple[Dict[str, Any], str]]:
    """Execute multiple tool calls concurrently."""
    if not tool_calls:
        return []
    
    logger.info(f"Executing {len(tool_calls)} tools concurrently")
    start_time = time.time()
    
    # Create async tasks for all tool calls
    tasks = [execute_tool_async(tool_call) for tool_call in tool_calls]
    
    # Execute all tasks concurrently
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    execution_time = time.time() - start_time
    logger.info(f"Concurrent tool execution completed in {execution_time:.2f}s")
    
    # Pair results with their tool calls
    tool_results = []
    for tool_call, result in zip(tool_calls, results):
        if isinstance(result, Exception):
            logger.error(f"Tool {tool_call.get('name')} failed: {result}")
            result_str = f"Tool execution failed: {str(result)}"
        else:
            result_str = result
        tool_results.append((tool_call, result_str))
    
    return tool_results

# Define the agent function with concurrent tool execution
def agent_function(state: AgentState) -> AgentState:
    """Main agent function that processes user input and generates responses.

    Runs a small loop with CONCURRENT tool execution for better performance.
    """
    messages = state["messages"]
    logger.info(f"Agent processing {len(messages)} messages")
    start_time = time.time()

    max_steps = 6
    for step in range(max_steps):
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not getattr(response, "tool_calls", None):
            # We have a final answer
            break

        # Execute ALL tool calls concurrently instead of sequentially
        tool_calls = response.tool_calls
        logger.info(f"Step {step + 1}: Executing {len(tool_calls)} tools concurrently")
        
        try:
            # Run concurrent tool execution
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop in a thread if current loop is running
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, execute_tools_concurrently(tool_calls))
                    tool_results = future.result(timeout=180)  # 3 minutes timeout
            else:
                tool_results = loop.run_until_complete(execute_tools_concurrently(tool_calls))
            
            # Add tool results as ToolMessages
            for tool_call, tool_result in tool_results:
                # Handle auto-triggered research for unsupported equipment
                if tool_call.get("name") == "check_lab_capability":
                    try:
                        parsed = json.loads(tool_result)
                        if isinstance(parsed, dict) and not parsed.get("supported"):
                            # Auto-trigger research - but this should be handled by the LLM in the next step
                            pass
                    except Exception:
                        pass
                
                # Echo tool results as ToolMessage so UI can show progress
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))
                
        except Exception as e:
            logger.error(f"Concurrent tool execution failed: {e}")
            # Fallback to sequential execution
            for tool_call in tool_calls:
                name = tool_call.get("name")
                args = tool_call.get("args", {}) or {}

                tool_result: str
                if name == "get_current_time":
                    tool_result = get_current_time.invoke({})
                elif name == "google_search":
                    tool_result = google_search.invoke({"query": args.get("query", "")})
                elif name == "parse_equipment_details":
                    tool_result = parse_equipment_details.invoke({"user_text": args.get("user_text", "")})
                elif name == "check_lab_capability":
                    tool_result = check_lab_capability.invoke({
                        "manufacturer": args.get("manufacturer", ""),
                        "model": args.get("model", ""),
                    })
                elif name == "parse_update_request":
                    tool_result = parse_update_request.invoke({
                        "request_text": args.get("request_text", "")
                    })
                elif name == "apply_update":
                    tool_result = apply_update.invoke({
                        "manufacturer": args.get("manufacturer", ""),
                        "model": args.get("model", ""),
                        "new_accredited": args.get("new_accredited", 0),
                        "user_name": args.get("user_name", ""),
                        "password": args.get("password", ""),
                    })
                else:
                    tool_result = f"Tool '{name}' not recognized."

                # Echo tool results as ToolMessage so UI can show progress
                messages.append(ToolMessage(
                    content=tool_result,
                    tool_call_id=tool_call["id"]
                ))

        # Continue loop to let the model consume tool outputs

    execution_time = time.time() - start_time
    logger.info(f"Agent completed processing in {execution_time:.2f}s after {step + 1} steps")
    
    return {"messages": messages}

# Create the graph
def create_agent_graph():
    """Create the LangGraph agent workflow."""
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("agent", agent_function)
    
    # Set the entry point
    workflow.set_entry_point("agent")
    
    # Add edges
    workflow.add_edge("agent", END)
    
    # Compile the graph
    return workflow.compile()

# Initialize the agent
agent = create_agent_graph()


# --------------
# Fallback capability inference agent
# --------------

def _load_lab_functions() -> Set[str]:
    global _LAB_FUNCTION_AREAS
    if _LAB_FUNCTION_AREAS:
        return _LAB_FUNCTION_AREAS
    try:
        path = os.path.join(os.path.dirname(__file__), "Tescom_Functions.json")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        areas: Set[str] = set()
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    for item in v:
                        if isinstance(item, str) and item.strip():
                            areas.add(item.strip().lower())
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, str) and item.strip():
                    areas.add(item.strip().lower())
        _LAB_FUNCTION_AREAS = areas
    except Exception:
        _LAB_FUNCTION_AREAS = set()
    return _LAB_FUNCTION_AREAS


def _equipment_type_from_text(text: str) -> Optional[str]:
    tokens = set(_tokenize_simple(text))
    
    # Check for explicit equipment type mentions first (high confidence)
    text_lower = text.lower()
    explicit_types = {
        "multimeter": ["multimeter", "dmm", "digital multimeter"],
        "oscilloscope": ["oscilloscope", "scope", "dso"],
        "power_supply": ["power supply", "psu"],
        "calibrator": ["calibrator", "calibration standard"],
        "spectrum_analyzer": ["spectrum analyzer", "signal analyzer"],
        "network_analyzer": ["network analyzer", "vna"],
        "signal_generator": ["signal generator", "function generator", "awg"]
    }
    
    for eq_type, keywords in explicit_types.items():
        if any(keyword in text_lower for keyword in keywords):
            return eq_type
    
    # If no explicit type found, use token-based matching
    keywords = {
        "multimeter": {"dmm", "multimeter", "voltage", "current", "resistance", "handheld", "digital", "true", "rms"},
        "oscilloscope": {"scope", "oscilloscope", "tds", "dso", "waveform", "bandwidth"},
        "calibrator": {"calibrator", "multi", "source", "5520a", "5720a", "standard"},
        "power_supply": {"psu", "supply", "dc", "power"},
        "spectrum_analyzer": {"spectrum", "analyzer", "rsa", "sa", "frequency"},
        "signal_generator": {"generator", "sg", "rf", "awgn", "awg", "signal"},
        "network_analyzer": {"vna", "network", "analyzer", "parameters"},
    }
    
    best = None
    best_score = 0
    for k, vocab in keywords.items():
        score = _jaccard(tokens, set(vocab))
        if score > best_score:
            best_score = score
            best = k
    
    # Special case: if we see model numbers that look like multimeters, bias toward multimeter
    if re.search(r'\b\d{3,4}[A-Za-z]?\b', text) and any(token in tokens for token in ['voltage', 'current', 'resistance', 'handheld', 'portable']):
        return "multimeter"
    
    return best


@tool
def infer_capability_via_research(query: str) -> str:
    """When a model is not found in the DB, research the asset and infer likely support.
    Steps:
      1) Use web search tool to gather a short spec snippet.
      2) Identify equipment type and compare to Tescom measurement areas from Tescom_Functions.json.
      3) Compare against known DB models by collapsed model token similarity.
      4) Return a conservative JSON: {likely_supported: bool, rationale, overlaps: [areas], similar_models: [...]}.
    """
    try:
        # 1) Web search with query expansion and vendor domain preference (PARALLEL)
        parsed_try = json.loads(parse_equipment_details.invoke({"user_text": query}))
        manu_guess = parsed_try.get("manufacturer") or ""
        model_guess = parsed_try.get("model") or ""
        
        # If no model was parsed, try to extract it heuristically from the original query
        if not model_guess:
            import re
            # Look for common model patterns like "394B", "34401A", etc.
            model_matches = re.findall(r'\b([A-Za-z]*\d+[A-Za-z]*)\b', query.upper())
            if model_matches:
                # Take the longest match (most likely to be a model number)
                model_guess = max(model_matches, key=len)
                logger.info(f"Extracted model '{model_guess}' from query using heuristics")
        
        queries = _expand_search_queries(manu_guess, model_guess, query)

        preferred = _preferred_domains_for_manu(manu_guess)
        description = ""
        snippet = ""
        
        # Parallel search execution
        async def run_parallel_searches():
            search_queries = queries[:4]  # Limit to 4 parallel searches
            search_tasks = [_async_google_search(q) for q in search_queries]
            
            try:
                results = await asyncio.gather(*search_tasks, return_exceptions=True)
                combined_snippet = ""
                best_description = ""
                
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        logger.warning(f"Search failed for query '{search_queries[i]}': {result}")
                        continue
                    
                    s = result if isinstance(result, str) else str(result)
                    combined_snippet += "\n" + s
                    
                    # Try to extract a snippet near a preferred-domain link
                    near = _extract_snippet_near_link(s, preferred)
                    if near and not best_description:
                        best_description = _extract_description_from_search(near)
                    
                    # Fallback to general extraction
                    if not best_description:
                        best_description = best_description or _extract_description_from_search(s)
                
                return best_description, combined_snippet
            except Exception as e:
                logger.error(f"Parallel search failed: {e}")
                # Fallback to single search
                fallback_result = await _async_google_search(queries[0] if queries else query)
                return _extract_description_from_search(fallback_result), fallback_result
        
        # Execute parallel searches
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Create a new event loop in a thread if current loop is running
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, run_parallel_searches())
                    description, snippet = future.result(timeout=120)  # 2 minutes timeout
            else:
                description, snippet = loop.run_until_complete(run_parallel_searches())
        except Exception as e:
            logger.error(f"Failed to run parallel searches: {e}")
            # Fallback to synchronous search
            sr = google_search.invoke({"query": queries[0] if queries else query})
            snippet = sr if isinstance(sr, str) else str(sr)
            description = _extract_description_from_search(snippet)
            
        functions_summary = _summarize_measurement_functions(snippet)

        # 2) Determine equipment type and overlaps with lab functions
        areas = _load_lab_functions()
        eq_type = _equipment_type_from_text(snippet + " " + query) or "unknown"
        tokens = set(_tokenize_simple(snippet + " " + query))
        overlaps = []
        # Direct keyword alignments for common categories to ensure coverage (e.g., oscilloscope)
        direct_map = {
            "oscilloscope": ["oscilloscope", "scope", "dso"],
            "multimeter": ["multimeter", "dmm", "true rms", "handheld", "digital multimeter", "voltage", "current", "resistance"],
            "calibrator": ["calibrator", "calibration standard"],
            "power supply": ["power supply", "psu"],
            "spectrum analyzer": ["spectrum analyzer", "rsa"],
            "signal generator": ["signal generator", "awg", "rf generator"],
            "network analyzer": ["network analyzer", "vna"],
        }
        for area in areas:
            area_tokens = set(_tokenize_simple(area))
            jaccard_score = _jaccard(tokens, area_tokens)
            
            # Standard jaccard threshold
            if jaccard_score >= 0.2:
                overlaps.append(area)
            # Special handling for multimeter capabilities
            elif eq_type == "multimeter":
                # Multimeters typically do voltage, current, and resistance measurements
                area_lower = area.lower()
                if ("voltage" in tokens and ("voltage" in area_lower and "measure" in area_lower)) or \
                   ("current" in tokens and ("current" in area_lower and "measure" in area_lower)) or \
                   ("resistance" in tokens and ("resistance" in area_lower and "measure" in area_lower)):
                    overlaps.append(area)
            else:
                # Also consider direct keyword match to measurement area description
                for label, kws in direct_map.items():
                    if any(k in tokens for k in _tokenize_simple(" ".join(kws))) and label in area.lower():
                        overlaps.append(area)
                        break

        # 3) Similar models from DB (by collapsed token containment)
        _load_equipment_data()
        collapsed_q = _collapse_alnum(query)
        similar: List[Dict[str, Any]] = []
        for recs in _MODEL_TO_RECORDS.values():
            for rec in recs:
                model = rec.get("model_num") or ""
                if not model:
                    continue
                if _collapse_alnum(model).startswith(collapsed_q) or collapsed_q in _collapse_alnum(model):
                    similar.append({
                        "manufacturer": rec.get("manufacturer"),
                        "model": rec.get("model_num"),
                        "description": rec.get("gage_descr"),
                        "accredited": int(rec.get("accredited", 0)),
                    })
        # Shorten list
        if len(similar) > 10:
            similar = similar[:10]

        # 4) Heuristic decision and advisory text
        likely_supported = bool(overlaps) or bool(similar)
        rationale_parts = []
        if overlaps:
            rationale_parts.append(f"Areas overlap: {', '.join(sorted(set(overlaps)))}")
        if similar:
            rationale_parts.append(f"Found {len(similar)} similar model(s) in DB")
        if not rationale_parts:
            rationale_parts.append("Insufficient overlap; recommend manual confirmation")

        # Extract query manufacturer/model for messaging
        try:
            p = json.loads(parse_equipment_details.invoke({"user_text": query}))
            q_manu = (p.get("manufacturer") or "").strip()
            q_model = (p.get("model") or "").strip()
        except Exception:
            q_manu, q_model = "", ""

        # Construct advice text (include description and functions summary in rationale)
        advice: str
        summary: str
        support_likely: bool = bool(likely_supported)
        if likely_supported:
            if similar:
                sim = similar[0]
                sim_str = f"{sim.get('manufacturer','').strip()} {sim.get('model','').strip()}".strip()
                if q_manu or q_model:
                    subject = f"{(q_manu + ' ' + q_model).strip()}".strip()
                else:
                    subject = query.strip()
                advice = (
                    f"The asset {subject} appears similar to {sim_str} in Tescom's supported database and is likely to be supported. "
                    f"Description: {description or '(no description found)'} "
                    f"Functions: {functions_summary} "
                    "Please confirm with the Lab Manager."
                )
                areas_str = ", ".join(sorted(set(overlaps))) if overlaps else ""
                if areas_str:
                    summary = f"Overlaps with measurement areas: {areas_str}. Calibration support is likely. Similar to {sim_str}."
                else:
                    summary = f"Calibration support is likely. Similar to {sim_str}."
            else:
                areas_str = ", ".join(sorted(set(overlaps))) if overlaps else "related areas"
                subject = (q_manu + " " + q_model).strip() or query.strip()
                advice = (
                    f"The asset {subject} overlaps with Tescom's measurement areas ({areas_str}) and is likely to be supported. "
                    f"Description: {description or '(no description found)'} "
                    f"Functions: {functions_summary} "
                    "Please confirm with the Lab Manager."
                )
                summary = f"Overlaps with measurement areas: {areas_str}. Calibration support is likely."
        else:
            subject = (q_manu + " " + q_model).strip() or query.strip()
            advice = (
                f"The asset {subject} does not match Tescom's measurement areas and no similar supported models were found; "
                f"Description: {description or '(no description found)'} "
                f"Functions: {functions_summary} "
                "it is not likely supported. Please confirm with the Lab Manager."
            )
            summary = "No overlaps or similar supported models found. Calibration support is unlikely."

        return json.dumps({
            "likely_supported": bool(likely_supported),
            "support_likely": support_likely,
            "equipment_type": eq_type,
            "overlaps": overlaps,
            "similar_models": similar,
            "rationale": "; ".join(rationale_parts),
            "description": description,
            "functions_summary": functions_summary,
            "advice": advice,
            "summary": summary,
            "query": {"manufacturer": q_manu or None, "model": q_model or None},
        })
    except Exception as e:
        return json.dumps({"error": str(e)})

# Async version of chat function for streaming support
async def chat_with_agent_async(user_input: str, conversation_history: list = None) -> tuple[str, list]:
    """
    Async version of chat with agent for better performance and streaming support.
    """
    if conversation_history is None:
        conversation_history = []
    
    logger.info(f"Async chat session started with input: {user_input[:50]}...")
    start_time = time.time()

    # System instruction guiding tool use and behavior
    system_prompt = SystemMessage(content=(
        "You are a metrology lab capabilities assistant and orchestrator."
        " Primary intents: \n"
        " - Capability check: parse equipment and check against Tescom list via tools.\n"
        " - Database update: only proceed when the user explicitly asks to 'update database' or equivalent.\n"
        "Update flow: Confirm intent, ask what to update, call 'parse_update_request' to extract manufacturer, model, and new level. Then ask for the user's name and the password. Only after getting both, pass values to 'apply_update'.\n"
        "Password is required and case-insensitive ('bigbrain'). Keep questions concise.\n"
        "When the capability check returns supported=false and includes 'fallback_research', ALWAYS include the following in your reply: a short 'Description:' using fallback_research.description, and a 'Likely Measurement Functions:' sentence using fallback_research.functions_summary. Also include the overlaps list from fallback_research.overlaps if present, and clearly state whether calibration support is likely based on fallback_research.summary. "
        "IMPORTANT: If fallback_research.likely_supported is True AND there are overlaps in fallback_research.overlaps, then calibration support is LIKELY and you should emphasize this positive conclusion."
    ))

    prior_messages = conversation_history
    if not any(isinstance(m, SystemMessage) for m in conversation_history):
        prior_messages = [system_prompt] + conversation_history

    state = {
        "messages": prior_messages + [HumanMessage(content=user_input)]
    }

    # Run the graph-backed agent (concurrent tool execution happens inside agent_function)
    result = agent.invoke(state)

    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    if ai_messages:
        response = ai_messages[-1].content
    else:
        response = "I apologize, but I couldn't generate a response."

    execution_time = time.time() - start_time
    logger.info(f"Async chat session completed in {execution_time:.2f}s, response length: {len(response)} chars")

    return response, result["messages"]

def chat_with_agent(user_input: str, conversation_history: list = None) -> tuple[str, list]:
    """
    Chat with the agent using the provided input. Uses an LLM orchestrator in front
    of the runtime tools to collect missing information and present results.
    
    Now uses async execution for better performance.
    """
    try:
        # Try to run asynchronously for better performance
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Create a new event loop in a thread if current loop is running
            with ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, chat_with_agent_async(user_input, conversation_history))
                return future.result(timeout=300)  # 5 minutes timeout
        else:
            return loop.run_until_complete(chat_with_agent_async(user_input, conversation_history))
    except Exception as e:
        logger.error(f"Async chat failed, falling back to sync: {e}")
        # Fallback to synchronous version if async fails
        return chat_with_agent_sync(user_input, conversation_history)

def chat_with_agent_sync(user_input: str, conversation_history: list = None) -> tuple[str, list]:
    """
    Synchronous fallback version of chat with agent.
    """
    if conversation_history is None:
        conversation_history = []
    
    logger.info(f"Sync chat session started with input: {user_input[:50]}...")
    start_time = time.time()

    # System instruction guiding tool use and behavior
    system_prompt = SystemMessage(content=(
        "You are a metrology lab capabilities assistant and orchestrator."
        " Primary intents: \n"
        " - Capability check: parse equipment and check against Tescom list via tools.\n"
        " - Database update: only proceed when the user explicitly asks to 'update database' or equivalent.\n"
        "Update flow: Confirm intent, ask what to update, call 'parse_update_request' to extract manufacturer, model, and new level. Then ask for the user's name and the password. Only after getting both, pass values to 'apply_update'.\n"
        "Password is required and case-insensitive ('bigbrain'). Keep questions concise.\n"
        "When the capability check returns supported=false and includes 'fallback_research', ALWAYS include the following in your reply: a short 'Description:' using fallback_research.description, and a 'Likely Measurement Functions:' sentence using fallback_research.functions_summary. Also include the overlaps list from fallback_research.overlaps if present, and clearly state whether calibration support is likely based on fallback_research.summary. "
        "IMPORTANT: If fallback_research.likely_supported is True AND there are overlaps in fallback_research.overlaps, then calibration support is LIKELY and you should emphasize this positive conclusion."
    ))

    prior_messages = conversation_history
    if not any(isinstance(m, SystemMessage) for m in conversation_history):
        prior_messages = [system_prompt] + conversation_history

    state = {
        "messages": prior_messages + [HumanMessage(content=user_input)]
    }

    # Run the graph-backed agent (multi-step tool loop inside agent_function)
    result = agent.invoke(state)

    ai_messages = [msg for msg in result["messages"] if isinstance(msg, AIMessage)]
    if ai_messages:
        response = ai_messages[-1].content
    else:
        response = "I apologize, but I couldn't generate a response."

    execution_time = time.time() - start_time
    logger.info(f"Sync chat session completed in {execution_time:.2f}s, response length: {len(response)} chars")

    return response, result["messages"]

if __name__ == "__main__":
    # Test the agent
    print("Agent initialized successfully!")
    print("You can now run the Gradio app with: python app.py")
