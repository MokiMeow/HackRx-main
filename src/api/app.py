from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple
import asyncio
from cachetools import TTLCache
import time
import logging
import os
import re
from dotenv import load_dotenv
import hashlib

# Document Processing
import requests
import fitz  # PyMuPDF
from openai import OpenAI
import pdfplumber
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load environment variables
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress verbose logs
for logger_name in ['pdfminer', 'urllib3', 'sentence_transformers', 'httpx']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

app = FastAPI(
    title="HackRx Ultimate API - GPT-4o",
    docs_url="/docs",
    redoc_url="/redoc"
)
security = HTTPBearer()

# Initialize models once
try:
    semantic_model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Semantic model loaded")
except:
    semantic_model = None


class OptimizedDocumentParser:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/pdf,*/*'
        })
    
    async def parse_document(self, doc_url: str) -> Dict:
        """Fast and comprehensive document parsing with retry logic"""
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Parsing document (attempt {attempt + 1}): {doc_url}")
                
                # Download PDF with timeout
                response = self.session.get(doc_url, timeout=90)
                response.raise_for_status()
                pdf_content = response.content
                
                if not pdf_content.startswith(b'%PDF'):
                    raise Exception("Invalid PDF format")
                
                logger.info(f"Downloaded {len(pdf_content)} bytes")
                
                # Parse with both methods for completeness
                full_text = ""
                tables = []
                
                # PyMuPDF - Fast and reliable
                try:
                    doc = fitz.open(stream=pdf_content, filetype="pdf")
                    for page_num, page in enumerate(doc):
                        text = page.get_text()
                        if text.strip():
                            full_text += f"\n[PAGE {page_num + 1}]\n{text}\n"
                    doc.close()
                    logger.info(f"PyMuPDF: {len(full_text)} chars")
                except Exception as e:
                    logger.warning(f"PyMuPDF error: {e}")
                
                # PDFPlumber - Better for tables
                try:
                    import io
                    with pdfplumber.open(io.BytesIO(pdf_content)) as pdf:
                        plumber_text = ""
                        for page_num, page in enumerate(pdf.pages):
                            # Extract text
                            text = page.extract_text()
                            if text:
                                plumber_text += f"\n[PAGE {page_num + 1}]\n{text}\n"
                            
                            # Extract tables
                            page_tables = page.extract_tables()
                            if page_tables:
                                for idx, table in enumerate(page_tables):
                                    # Add table to text for searchability
                                    table_text = f"\n[TABLE P{page_num + 1}-{idx + 1}]\n"
                                    for row in table:
                                        if row:
                                            clean_row = [str(cell).strip() if cell else "" for cell in row]
                                            table_text += " | ".join(clean_row) + "\n"
                                    plumber_text += table_text
                                    
                                    # Store table
                                    tables.append({
                                        'page': page_num + 1,
                                        'data': table
                                    })
                        
                        # Use plumber text if it's longer (usually more complete)
                        if len(plumber_text) > len(full_text):
                            full_text = plumber_text
                        
                        logger.info(f"PDFPlumber: {len(plumber_text)} chars, {len(tables)} tables")
                except Exception as e:
                    logger.warning(f"PDFPlumber error: {e}")
                
                if not full_text:
                    raise Exception("No text extracted from PDF")
                
                # Clean text minimally
                full_text = re.sub(r'\n{4,}', '\n\n\n', full_text)
                full_text = re.sub(r' {3,}', '  ', full_text)
                full_text = re.sub(r'Rs\.?\s*(\d)', r'Rs.\1', full_text)
                
                logger.info(f"Parsed successfully: {len(full_text)} chars")
                
                return {
                    'full_text': full_text,
                    'tables': tables
                }
                
            except Exception as e:
                logger.error(f"Parse attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {'full_text': '', 'tables': []}


class SmartRetriever:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.chunk_cache = {}
    
    def index_documents(self, chunks: List[Dict]):
        """Index chunks efficiently with error handling"""
        self.chunks = chunks
        
        if not chunks:
            return
        
        texts = [c['text'] for c in chunks]
        
        # Create embeddings with retry logic
        if semantic_model and len(chunks) < 2000:
            for attempt in range(3):
                try:
                    self.embeddings = semantic_model.encode(
                        texts,
                        batch_size=64,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=True
                    )
                    logger.info(f"Created embeddings for {len(chunks)} chunks")
                    break
                except Exception as e:
                    logger.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                    if attempt == 2:
                        self.embeddings = None
        
        # Create TF-IDF with error handling
        try:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                stop_words='english',
                min_df=1,
                max_df=0.95
            )
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
            logger.info("Created TF-IDF matrix")
        except Exception as e:
            logger.warning(f"TF-IDF failed: {e}")
            self.tfidf_matrix = None
            self.tfidf_vectorizer = None
    
    def retrieve(self, query: str, top_k: int = 15) -> List[Dict]:
        """Fast and accurate retrieval with enhanced scoring"""
        if not self.chunks:
            return []
        
        # Check cache
        cache_key = hashlib.md5(query.encode()).hexdigest()
        if cache_key in self.chunk_cache:
            return self.chunk_cache[cache_key][:top_k]
        
        query_lower = query.lower()
        scores = []
        
        # Extract query features once
        query_words = set(query_lower.split())
        query_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        query_amounts = re.findall(r'[₹$]\s*[\d,]+(?:\.\d+)?|Rs\.?\s*[\d,]+', query)
        
        # Pre-compute embeddings
        query_embedding = None
        if self.embeddings is not None and semantic_model:
            try:
                query_embedding = semantic_model.encode([query], show_progress_bar=False, normalize_embeddings=True)[0]
            except:
                pass
        
        # Pre-compute TF-IDF
        query_tfidf = None
        if self.tfidf_vectorizer and self.tfidf_matrix is not None:
            try:
                query_tfidf = self.tfidf_vectorizer.transform([query])
            except:
                pass
        
        # Score all chunks
        for i, chunk in enumerate(self.chunks):
            chunk_text = chunk['text']
            chunk_lower = chunk_text.lower()
            
            score = 0
            
            # 1. Exact substring match (highest priority)
            if len(query) > 15:
                if query_lower in chunk_lower:
                    score += 100
                else:
                    # Check for most words present
                    words_found = sum(1 for w in query_words if w in chunk_lower)
                    if words_found >= len(query_words) * 0.8:
                        score += 50
            
            # 2. Critical number matching
            if query_numbers:
                numbers_found = sum(1 for num in query_numbers if num in chunk_text)
                if numbers_found == len(query_numbers):
                    score += 80
                elif numbers_found > 0:
                    score += (numbers_found * 30)
            
            # 3. Amount matching (for financial queries)
            if query_amounts:
                amounts_found = sum(1 for amt in query_amounts if amt in chunk_text)
                score += amounts_found * 40
            
            # 4. Word overlap scoring
            chunk_words = set(chunk_lower.split())
            overlap = len(query_words & chunk_words)
            if query_words:
                score += (overlap / len(query_words)) * 30
            
            # 5. Semantic similarity
            if query_embedding is not None and self.embeddings is not None:
                try:
                    similarity = np.dot(query_embedding, self.embeddings[i])
                    score += similarity * 25
                except:
                    pass
            
            # 6. TF-IDF similarity
            if query_tfidf is not None and self.tfidf_matrix is not None:
                try:
                    tfidf_sim = cosine_similarity(query_tfidf, self.tfidf_matrix[i:i+1])[0][0]
                    score += tfidf_sim * 25
                except:
                    pass
            
            # 7. Table bonus for relevant queries
            if '[TABLE' in chunk_text:
                table_keywords = ['table', 'limit', 'coverage', 'amount', 'plan', 'variant', 'sub-limit', 'maximum']
                if any(kw in query_lower for kw in table_keywords):
                    score += 30
            
            # 8. Context proximity bonus (words near each other)
            if len(query_words) > 1:
                # Check if query words appear close together
                for word in query_words:
                    if word in chunk_lower:
                        word_pos = chunk_lower.find(word)
                        nearby_words = sum(1 for w in query_words if w in chunk_lower[max(0, word_pos-100):word_pos+100])
                        if nearby_words > 1:
                            score += nearby_words * 5
                            break
            
            if score > 0:
                scores.append({
                    'text': chunk_text,
                    'score': score,
                    'metadata': chunk.get('metadata', {})
                })
        
        # Sort by score
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        # Cache results
        self.chunk_cache[cache_key] = scores
        
        # Log scores
        if scores:
            logger.info(f"Top scores: {[round(s['score'], 2) for s in scores[:3]]}")
        
        return scores[:top_k]


class GPT4oReasoningEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.answer_cache = {}
        
        # Use GPT-4o - the best model for accuracy + speed + cost
        self.model = 'gpt-4o'
        logger.info(f"Using model: {self.model} (Best for accuracy + speed + cost)")
    
    async def reason(self, question: str, context: List[Dict], tables: List = None) -> Dict:
        """Generate accurate answer with GPT-4o and retry logic"""
        # Check cache
        cache_key = hashlib.md5(f"{question}:{context[0]['text'][:100] if context else ''}".encode()).hexdigest()
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Use more context for better accuracy (12 chunks)
                context_parts = []
                for i, chunk in enumerate(context[:12]):
                    score = chunk.get('score', 0)
                    context_parts.append(f"[Context {i+1} - Score: {score:.1f}]\n{chunk['text']}\n")
                
                # Add relevant tables if available
                table_context = ""
                if tables and any(kw in question.lower() for kw in ['table', 'limit', 'amount', 'coverage', 'plan']):
                    table_context = "\n\nRELEVANT TABLES:\n"
                    for table in tables[:3]:
                        if table.get('data'):
                            table_context += f"\n[Table Page {table.get('page', 'N/A')}]\n"
                            for row in table['data'][:15]:
                                if row:
                                    table_context += " | ".join([str(c) if c else "" for c in row]) + "\n"
                
                full_context = "\n---\n".join(context_parts) + table_context
                
                # Enhanced prompt for maximum accuracy with GPT-4o
                prompt = f"""You are analyzing a document to answer a question with 100% accuracy.

DOCUMENT CONTENT (sorted by relevance):
{full_context}

QUESTION: {question}

CRITICAL REQUIREMENTS FOR ACCURACY:
1. Read ALL provided context carefully - the answer may be spread across multiple sections
2. NEVER approximate, round, or guess any values - use exact numbers from the document
3. Include ALL relevant details: every number, date, percentage, amount, condition
4. If there are multiple values/plans/options, list them ALL with their specific details
5. If information appears in tables, preserve that structure and include all values
6. Quote specific values directly - don't paraphrase numbers or percentages
7. Include ALL conditions, exceptions, waiting periods, and special cases
8. If different sections provide related information, combine them comprehensively
9. Use proper units (days, months, years, Rs., %, etc.) exactly as in the document
10. If the specific information is NOT in the document, state: "This information is not found in the provided document."

ANSWER FORMAT:
- Start with the direct, complete answer
- Use bullet points or lists for multiple items
- Include all numerical values with their units
- Be comprehensive - don't leave out any relevant detail
- Organize information clearly but include everything

Answer:"""

                # API call with GPT-4o
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a meticulous document analyst. Extract and provide COMPLETE information from documents. Include ALL numbers, conditions, and details. Never summarize or skip information."
                        },
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=500,  # Enough for detailed answers
                    temperature=0.0,  # Zero for consistency
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0
                )
                
                answer = response.choices[0].message.content.strip()
                
                # Calculate confidence
                avg_score = sum(c.get('score', 0) for c in context[:3]) / 3 if context else 0
                confidence = min(0.99, avg_score / 50)
                
                result = {
                    'answer': answer,
                    'confidence': confidence
                }
                
                # Cache result
                self.answer_cache[cache_key] = result
                
                return result
                
            except Exception as e:
                logger.error(f"GPT-4o reasoning attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)  # Wait before retry
                else:
                    return {
                        'answer': f"Error after {max_retries} attempts: {str(e)}",
                        'confidence': 0.0
                    }


def create_smart_chunks(doc_data: Dict) -> List[Dict]:
    """Create optimized chunks for maximum retrieval accuracy"""
    chunks = []
    full_text = doc_data.get('full_text', '')
    
    if not full_text:
        return chunks
    
    # Multiple chunk sizes for comprehensive coverage
    configs = [
        {'size': 2000, 'overlap': 500},  # Large chunks for context
        {'size': 1200, 'overlap': 300},  # Medium chunks
        {'size': 600, 'overlap': 150},   # Small chunks for specific info
    ]
    
    for config in configs:
        size, overlap = config['size'], config['overlap']
        
        pos = 0
        while pos < len(full_text):
            end = min(pos + size, len(full_text))
            chunk_text = full_text[pos:end]
            
            # Try to end at sentence or paragraph boundary
            if end < len(full_text):
                # Try paragraph first
                last_para = chunk_text.rfind('\n\n')
                if last_para > size * 0.7:
                    chunk_text = chunk_text[:last_para]
                else:
                    # Try sentence
                    last_period = chunk_text.rfind('. ')
                    if last_period > size * 0.7:
                        chunk_text = chunk_text[:last_period + 1]
            
            if len(chunk_text.strip()) > 100:
                chunks.append({
                    'text': chunk_text.strip(),
                    'metadata': {'start': pos, 'size': size}
                })
            
            pos += (size - overlap)
    
    # Add page-based chunks
    if '[PAGE' in full_text:
        pages = re.split(r'\[PAGE \d+\]', full_text)
        for i, page in enumerate(pages):
            if len(page.strip()) > 100:
                # Split very large pages
                if len(page) > 3000:
                    for j in range(0, len(page), 2500):
                        page_chunk = page[j:j+2500]
                        if len(page_chunk.strip()) > 100:
                            chunks.append({
                                'text': f"[PAGE {i+1} Part {j//2500 + 1}] {page_chunk.strip()}",
                                'metadata': {'type': 'page', 'page': i+1}
                            })
                else:
                    chunks.append({
                        'text': f"[PAGE {i+1}] {page.strip()}",
                        'metadata': {'type': 'page', 'page': i+1}
                    })
    
    # Add table chunks
    if '[TABLE' in full_text:
        tables = re.findall(r'(\[TABLE[^\]]*\][^[]{0,3000})', full_text)
        for i, table in enumerate(tables):
            if table.strip():
                chunks.append({
                    'text': table.strip(),
                    'metadata': {'type': 'table', 'index': i}
                })
    
    # Deduplicate
    seen = set()
    unique = []
    for chunk in chunks:
        # Use hash for better deduplication
        chunk_hash = hashlib.md5(chunk['text'].encode()).hexdigest()
        if chunk_hash not in seen:
            seen.add(chunk_hash)
            unique.append(chunk)
    
    logger.info(f"Created {len(unique)} unique chunks")
    return unique


# Initialize components
document_parser = OptimizedDocumentParser()
retriever = SmartRetriever()
reasoning_engine = GPT4oReasoningEngine()

# Global cache with larger size
response_cache = TTLCache(maxsize=200, ttl=7200)  # 2 hour TTL


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


@app.post("/hackrx/run")
async def process_query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Process queries with GPT-4o for maximum accuracy and speed"""
    
    start_time = time.time()
    
    # Validate token
    if credentials.credentials != os.getenv('BEARER_TOKEN', 'test-token'):
        raise HTTPException(status_code=401, detail="Invalid token")
    
    # Check cache
    doc_hash = hashlib.md5(request.documents.encode()).hexdigest()
    questions_hash = hashlib.md5(':'.join(request.questions).encode()).hexdigest()
    cache_key = f"{doc_hash}:{questions_hash}"
    
    if cache_key in response_cache:
        logger.info("Returning cached response")
        return response_cache[cache_key]
    
    try:
        logger.info(f"Processing {len(request.questions)} questions with GPT-4o")
        
        # Parse document with retry
        doc_data = await document_parser.parse_document(request.documents)
        
        if not doc_data.get('full_text'):
            return {"answers": ["Unable to parse document after multiple attempts"] * len(request.questions)}
        
        # Create comprehensive chunks
        chunks = create_smart_chunks(doc_data)
        if not chunks:
            return {"answers": ["No content found in document"] * len(request.questions)}
        
        # Index chunks
        retriever.index_documents(chunks)
        
        # Process questions
        answers = []
        for question in request.questions:
            logger.info(f"Q: {question}")
            
            # Retrieve with more chunks for better accuracy
            relevant = retriever.retrieve(question, top_k=20)
            
            if not relevant:
                answers.append("No relevant information found in the document for this question.")
                continue
            
            # Generate answer with GPT-4o
            result = await reasoning_engine.reason(
                question, 
                relevant,
                doc_data.get('tables', [])
            )
            
            answers.append(result['answer'])
            logger.info(f"Confidence: {result['confidence']:.1%}")
        
        # Cache response
        response = {"answers": answers}
        response_cache[cache_key] = response
        
        logger.info(f"Completed in {time.time() - start_time:.1f}s with GPT-4o")
        
        return response
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"answers": [f"System error: {str(e)}"] * len(request.questions)}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "service": "HackRx Ultimate API", 
        "model": "gpt-4o",
        "description": "Best model for accuracy + speed + cost"
    }


@app.get("/")
async def root():
    return {
        "message": "HackRx Ultimate API Running", 
        "model": "gpt-4o",
        "features": "Maximum accuracy with optimal speed and cost"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)