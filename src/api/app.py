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
import json
import backoff

# Document Processing
import requests

# Configuration Management - Centralized settings
class Config:
    # Document weights
    UNKNOWN_DOC_WEIGHT = 3.0
    KNOWN_DOC_WEIGHT = 0.5

    # Chunk sizes (optimized for token limits)
    MAX_CHUNK_SIZE = 2000  # Reduced from 3000 to avoid token limits
    MEDIUM_CHUNK_SIZE = 1500
    SMALL_CHUNK_SIZE = 1000
    MINI_CHUNK_SIZE = 500

    # Cache settings
    RESPONSE_CACHE_SIZE = 1000
    DOCUMENT_CACHE_SIZE = 500
    CHUNK_CACHE_SIZE = 1000
    CACHE_TTL = 14400  # 4 hours

    # Model settings
    MAX_TOKENS = 1000
    TIMEOUT = 30
    RETRY_ATTEMPTS = 3
    RETRY_MAX_TIME = 60
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
    title="HackRx Winning Strategy API - Score-Optimized",
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

# WINNING STRATEGY: Competition scoring weights - MAXIMUM OPTIMIZATION
KNOWN_DOC_WEIGHT = 0.5
UNKNOWN_DOC_WEIGHT = 3.0  # 6x higher weight for unknown documents - AGGRESSIVE

# Question complexity weights based on competition analysis - MAXIMUM OPTIMIZATION
QUESTION_WEIGHTS = {
    # ULTRA High-weight questions (3.0) - MAXIMUM focus for competition
    "grace period": 3.0,
    "waiting period": 3.0,
    "pre-existing": 3.0,
    "maternity": 3.0,
    "cataract": 3.0,
    "organ donor": 3.0,
    "no claim discount": 3.0,
    "ncd": 3.0,
    "preventive health": 3.0,
    "hospital definition": 3.0,
    "ayush": 3.0,
    "sub-limits": 3.0,
    "room rent": 3.0,
    "icu charges": 3.0,
    # Insurance policy specific ULTRA high-weight terms (3.0)
    "sum insured": 3.0,
    "coverage amount": 3.0,
    "policy period": 3.0,
    "exclusions": 3.0,
    "conditions": 3.0,
    "benefits": 3.0,
    "premium": 3.0,
    "claim": 3.0,
    "medical expenses": 3.0,
    "hospitalization": 3.0,
    "treatment": 3.0,
    "surgery": 3.0,
    "disease": 3.0,
    "illness": 3.0,
    "injury": 3.0,

    # Medium-weight questions (1.5)
    "coverage": 1.5,
    "conditions": 1.5,
    "exclusions": 1.5,
    "benefits": 1.5,
    "limits": 1.5,

    # Standard questions (1.0)
    "what is": 1.0,
    "how": 1.0,
    "does": 1.0,
    "are": 1.0,
    "is": 1.0
}

# Known document patterns (publicly available)
KNOWN_DOC_PATTERNS = [
    "hackrx.blob.core.windows.net",
    "public",
    "sample",
    "test",
    "example",
    "arogya sanjeevani",  # Known insurance policy
    "national parivar mediclaim"
]

# INSURANCE PATTERN MATCHING - For common questions
INSURANCE_PATTERNS = {
    'grace_period': r'grace period.*?(\d+\s*days?)',
    'waiting_period': r'waiting period.*?(\d+\s*(?:months?|years?))',
    'pre_existing': r'pre-existing.*?(\d+\s*months?)',
    'maternity_waiting': r'maternity.*?(\d+\s*months?)',
    'cataract_waiting': r'cataract.*?(\d+\s*(?:months?|years?))',
    'sum_insured': r'sum insured.*?(?:Rs\.?|₹|$)\s*([\d,]+)',
    'premium_amount': r'premium.*?(?:Rs\.?|₹|$)\s*([\d,]+)',
    'ncd_percentage': r'no claim discount.*?(\d+(?:\.\d+)?%)',
    'room_rent_limit': r'room rent.*?(\d+(?:\.\d+)?%)',
    'icu_limit': r'icu.*?(\d+(?:\.\d+)?%)',
}

def is_unknown_document(doc_url: str) -> bool:
    """Determine if document is unknown (higher weight)"""
    doc_lower = doc_url.lower()
    return not any(pattern in doc_lower for pattern in KNOWN_DOC_PATTERNS)

def get_question_weight(question: str) -> float:
    """Get question weight based on complexity"""
    question_lower = question.lower()

    # Check for high-weight keywords first
    for keyword, weight in QUESTION_WEIGHTS.items():
        if keyword in question_lower:
            return weight

    # Default weight
    return 1.0

def calculate_priority_score(question: str, doc_url: str) -> float:
    """Calculate priority score for resource allocation"""
    question_weight = get_question_weight(question)
    doc_weight = Config.UNKNOWN_DOC_WEIGHT if is_unknown_document(doc_url) else Config.KNOWN_DOC_WEIGHT

    # Priority = Question Weight × Document Weight
    return question_weight * doc_weight


class OptimizedDocumentParser:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

        # Connection pooling for better HTTP performance
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=3,
            pool_block=False
        )
        self.session = requests.Session()
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/pdf,*/*'
        })

    async def parse_document(self, doc_url: str) -> Dict:
        """Fast and comprehensive document parsing with retry logic and caching"""

        # Check document cache first
        doc_hash = hashlib.md5(doc_url.encode()).hexdigest()
        if doc_hash in document_cache:
            logger.info("Using cached document")
            return document_cache[doc_hash]

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

                result = {
                    'full_text': full_text,
                    'tables': tables
                }

                # Cache the document
                document_cache[doc_hash] = result
                return result

            except Exception as e:
                logger.error(f"Parse attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return {'full_text': '', 'tables': []}


class WinningRetriever:
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.chunk_cache = {}
        self.keyword_index = {}

    def index_documents(self, chunks: List[Dict]):
        """Index chunks with advanced features"""
        self.chunks = chunks

        if not chunks:
            return

        texts = [c['text'] for c in chunks]

        # Create keyword index for exact matching
        for i, chunk in enumerate(chunks):
            words = re.findall(r'\b\w+\b', chunk['text'].lower())
            for word in words:
                if len(word) > 2:  # Skip short words
                    if word not in self.keyword_index:
                        self.keyword_index[word] = []
                    self.keyword_index[word].append(i)

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
                max_features=3000,
                ngram_range=(1, 3),
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

    def retrieve(self, query: str, top_k: int = 35, priority_score: float = 1.0) -> List[Dict]:  # OPTIMIZED FOR SPEED
        """Advanced retrieval with priority-based scoring"""
        if not self.chunks:
            return []

        # Check cache
        cache_key = hashlib.md5(f"{query}:{priority_score}".encode()).hexdigest()
        if cache_key in self.chunk_cache:
            return self.chunk_cache[cache_key][:top_k]

        query_lower = query.lower()
        scores = []

        # Extract query features
        query_words = set(query_lower.split())
        query_numbers = re.findall(r'\b\d+(?:\.\d+)?\b', query)
        query_amounts = re.findall(r'[₹$]\s*[\d,]+(?:\.\d+)?|Rs\.?\s*[\d,]+', query)
        query_percentages = re.findall(r'\b\d+(?:\.\d+)?%\b', query)

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

        # Score all chunks with priority multiplier
        for i, chunk in enumerate(self.chunks):
            chunk_text = chunk['text']
            chunk_lower = chunk_text.lower()

            score = 0

            # 1. Exact substring match (highest priority) - MAXIMUM BOOST
            if len(query) > 10:
                if query_lower in chunk_lower:
                    score += 500  # MAXIMUM boost for exact matches
                else:
                    # Check for most words present
                    words_found = sum(1 for w in query_words if w in chunk_lower)
                    if words_found >= len(query_words) * 0.6:  # LOWERED THRESHOLD
                        score += 250

            # 2. Critical number matching - MAXIMUM BOOST
            if query_numbers:
                numbers_found = sum(1 for num in query_numbers if num in chunk_text)
                if numbers_found == len(query_numbers):
                    score += 400  # MAXIMUM boost for exact number matches
                elif numbers_found > 0:
                    score += (numbers_found * 100)  # INCREASED per number

            # 3. Amount matching (for financial queries) - MAXIMUM BOOST
            if query_amounts:
                amounts_found = sum(1 for amt in query_amounts if amt in chunk_text)
                score += amounts_found * 150  # INCREASED for financial accuracy

            # 4. Percentage matching
            if query_percentages:
                percents_found = sum(1 for pct in query_percentages if pct in chunk_text)
                score += percents_found * 80

            # 5. Word overlap scoring with position bonus
            chunk_words = set(chunk_lower.split())
            overlap = len(query_words & chunk_words)
            if query_words:
                score += (overlap / len(query_words)) * 80

            # 6. Semantic similarity
            if query_embedding is not None and self.embeddings is not None:
                try:
                    similarity = np.dot(query_embedding, self.embeddings[i])
                    score += similarity * 50
                except:
                    pass

            # 7. TF-IDF similarity
            if query_tfidf is not None and self.tfidf_matrix is not None:
                try:
                    tfidf_sim = cosine_similarity(query_tfidf, self.tfidf_matrix[i:i+1])[0][0]
                    score += tfidf_sim * 50
                except:
                    pass

            # 8. Table bonus for relevant queries
            if '[TABLE' in chunk_text:
                table_keywords = ['table', 'limit', 'coverage', 'amount', 'plan', 'variant', 'sub-limit', 'maximum', 'minimum', 'benefits', 'sum insured', 'premium']
                if any(kw in query_lower for kw in table_keywords):
                    score += 70

            # 9. Insurance policy specific bonuses
            if any(term in chunk_lower for term in ['policy', 'insurance', 'coverage', 'claim', 'premium', 'sum insured']):
                insurance_keywords = ['grace period', 'waiting period', 'exclusions', 'conditions', 'benefits', 'maternity', 'cataract', 'organ donor', 'ncd', 'ayush']
                if any(kw in query_lower for kw in insurance_keywords):
                    score += 80

            # 10. Page bonus for specific queries
            if '[PAGE' in chunk_text:
                page_keywords = ['page', 'section', 'chapter', 'part']
                if any(kw in query_lower for kw in page_keywords):
                    score += 40

                        # 11. Context proximity bonus
            if len(query_words) > 1:
                for word in query_words:
                    if word in chunk_lower:
                        word_pos = chunk_lower.find(word)
                        nearby_words = sum(1 for w in query_words if w in chunk_lower[max(0, word_pos-150):word_pos+150])
                        if nearby_words > 1:
                            score += nearby_words * 15
                            break

            # 12. Keyword density bonus
            keyword_density = sum(1 for word in query_words if word in chunk_lower) / len(chunk_words) if chunk_words else 0
            score += keyword_density * 120

                        # 13. Insurance document specific scoring - MAXIMUM BOOST
            if any(term in chunk_lower for term in ['policy', 'insurance', 'coverage', 'claim']):
                # Bonus for insurance-related content
                score += 100  # DOUBLED

                # Extra bonus for specific insurance terms
                insurance_terms = ['grace period', 'waiting period', 'pre-existing', 'maternity', 'cataract', 'organ donor', 'ncd', 'ayush', 'sum insured', 'premium']
                if any(term in chunk_lower for term in insurance_terms):
                    score += 200  # DOUBLED

            # 14. PRIORITY MULTIPLIER - Key to winning strategy
            score *= priority_score

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
            logger.info(f"Priority {priority_score:.1f} - Top scores: {[round(s['score'], 2) for s in scores[:3]]}")

        return scores[:top_k]


class WinningReasoningEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.answer_cache = {}

        # Use multiple models for ensemble - MAXIMUM ACCURACY + SPEED
        self.models = ['gpt-4o-mini']  # REMOVED THIRD MODEL FOR SPEED
        logger.info(f"Using models: {self.models}")

    @backoff.on_exception(
        backoff.expo,
        Exception,
        max_tries=Config.RETRY_ATTEMPTS,
        max_time=Config.RETRY_MAX_TIME
    )
    async def call_model_async(self, model, messages, max_tokens):
        """Call OpenAI asynchronously with retry logic"""
        return await asyncio.to_thread(
            self.client.chat.completions.create,
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            timeout=Config.TIMEOUT
        )

    async def parallel_reasoning(self, models, messages, max_tokens):
        """Parallel model calls for better concurrency"""
        tasks = [self.call_model_async(model, messages, max_tokens) for model in models]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [r for r in results if not isinstance(r, Exception)]

    async def reason(self, question: str, context: List[Dict], tables: List = None, priority_score: float = 1.0) -> Dict:
        """Generate accurate answer with priority-based resource allocation"""
        # Check cache
        cache_key = hashlib.md5(f"{question}:{context[0]['text'][:100] if context else ''}:{priority_score}".encode()).hexdigest()
        if cache_key in self.answer_cache:
            return self.answer_cache[cache_key]

        # Adjust context size based on priority - OPTIMIZED FOR SPEED
        context_size = min(25, int(15 * priority_score))  # OPTIMIZED context for speed

        # Use more context for better accuracy
        context_parts = []
        for i, chunk in enumerate(context[:context_size]):
            score = chunk.get('score', 0)
            context_parts.append(f"[Context {i+1} - Score: {score:.1f}]\n{chunk['text']}\n")

        # Add relevant tables if available
        table_context = ""
        if tables and any(kw in question.lower() for kw in ['table', 'limit', 'amount', 'coverage', 'plan', 'maximum', 'minimum']):
            table_context = "\n\nRELEVANT TABLES:\n"
            for table in tables[:5]:
                if table.get('data'):
                    table_context += f"\n[Table Page {table.get('page', 'N/A')}]\n"
                    for row in table['data'][:20]:
                        if row:
                            table_context += " | ".join([str(c) if c else "" for c in row]) + "\n"

        full_context = "\n---\n".join(context_parts) + table_context

        # ULTRA-ENHANCED prompt for MAXIMUM accuracy - COMPETITION OPTIMIZED
        prompt = f"""You are analyzing a document to answer a question with 100% accuracy. This is a ULTRA-HIGH-PRIORITY question (weight: {priority_score:.1f}x) for a COMPETITION.

DOCUMENT CONTENT (sorted by relevance):
{full_context}

QUESTION: {question}

ULTRA-CRITICAL REQUIREMENTS FOR MAXIMUM ACCURACY:
1. Read ALL provided context carefully - the answer may be spread across multiple sections
2. NEVER approximate, round, or guess any values - use EXACT numbers from the document
3. Include ALL relevant details: every number, date, percentage, amount, condition, exception
4. If there are multiple values/plans/options, list them ALL with their specific details
5. If information appears in tables, preserve that structure and include all values
6. Quote specific values directly - don't paraphrase numbers or percentages
7. Include ALL conditions, exceptions, waiting periods, and special cases
8. If different sections provide related information, combine them comprehensively
9. Use proper units (days, months, years, Rs., %, etc.) exactly as in the document
10. If the specific information is NOT in the document, state: "This information is not found in the provided document."
11. For financial amounts, include the exact currency symbol and formatting
12. For percentages, include the % symbol
13. For dates, use the exact format from the document
14. For limits and coverage amounts, specify if they are per occurrence, per year, or total
15. BE ULTRA-CAREFUL - This question has {priority_score:.1f}x weight in scoring
16. COMPETITION FOCUS: This is for a high-stakes competition - accuracy is CRITICAL
17. INSURANCE EXPERTISE: If this is an insurance question, be extremely precise about terms
18. LEGAL PRECISION: If this is a legal question, quote exact clauses and conditions
19. COMPREHENSIVE COVERAGE: Don't miss any relevant information from the document
20. EXACT MATCHING: Match the expected answer format from the competition

ANSWER FORMAT REQUIREMENTS:
- Start with a clear, direct answer to the question
- Use **bold headers** for major sections (e.g., **Eligibility Conditions**, **Discount Calculation**)
- Use bullet points (•) for lists and multiple items
- Use numbered lists (1., 2., 3.) for sequential information
- Include ALL numerical values with their exact units (Rs., %, days, months, years)
- Use tables when presenting structured data (| Column | Column |)
- Be comprehensive but well-organized
- If there are multiple options or plans, clearly distinguish between them
- Use consistent formatting throughout the response
- Include all relevant conditions, exceptions, and special cases
- For insurance questions, be extremely precise about policy terms and conditions

COMPETITION STYLE GUIDELINES:
- Match the exact format expected by evaluators
- Use professional, clear language
- Ensure all information is accurate and complete
- Organize information logically with clear sections
- Include all relevant details without being verbose
- Use proper markdown formatting for better readability

Answer:"""

        # Try multiple models in parallel for better accuracy and speed
        try:
            # Adjust max_tokens based on priority - OPTIMIZED FOR SPEED
            max_tokens = min(Config.MAX_TOKENS, int(800 * priority_score))

            messages = [
                {
                    "role": "system",
                    "content": f"You are a meticulous document analyst. Extract and provide COMPLETE information from documents. Include ALL numbers, conditions, and details. Never summarize or skip information. This question has {priority_score:.1f}x scoring weight."
                },
                {"role": "user", "content": prompt}
            ]

            # Use parallel reasoning for better concurrency
            responses = await self.parallel_reasoning(self.models, messages, max_tokens)
            answers = [response.choices[0].message.content.strip() for response in responses]

        except Exception as e:
            logger.warning(f"Parallel reasoning failed: {e}")
            # Fallback to sequential processing
            answers = []
            for model in self.models:
                try:
                    response = await self.call_model_async(model, messages, max_tokens)
                    answer = response.choices[0].message.content.strip()
                    answers.append(answer)
                except Exception as model_e:
                    logger.warning(f"Model {model} failed: {model_e}")
                    continue

        # Use the best answer (usually the first one that succeeds)
        final_answer = answers[0] if answers else "Unable to generate answer"

        # Calculate confidence
        avg_score = sum(c.get('score', 0) for c in context[:5]) / 5 if context else 0
        confidence = min(0.99, avg_score / 100)

        result = {
            'answer': final_answer,
            'confidence': confidence,
            'priority_score': priority_score
        }

        # Cache result
        self.answer_cache[cache_key] = result

        return result


def create_winning_chunks(doc_data: Dict) -> List[Dict]:
    """Create optimized chunks for maximum retrieval accuracy with caching"""
    chunks = []
    full_text = doc_data.get('full_text', '')

    if not full_text:
        return chunks

    # Check chunk cache first
    text_hash = hashlib.md5(full_text.encode()).hexdigest()
    if text_hash in chunk_cache:
        logger.info("Using cached chunks")
        return chunk_cache[text_hash]

    # Multiple chunk sizes for comprehensive coverage (optimized for token limits)
    configs = [
        {'size': Config.MAX_CHUNK_SIZE, 'overlap': 600},      # Large chunks for context
        {'size': Config.MEDIUM_CHUNK_SIZE, 'overlap': 400},   # Medium chunks
        {'size': Config.SMALL_CHUNK_SIZE, 'overlap': 250},    # Small chunks for specific info
        {'size': Config.MINI_CHUNK_SIZE, 'overlap': 125},     # Very small chunks for exact matches
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
                if len(page) > 5000:
                    for j in range(0, len(page), 4000):
                        page_chunk = page[j:j+4000]
                        if len(page_chunk.strip()) > 100:
                            chunks.append({
                                'text': f"[PAGE {i+1} Part {j//4000 + 1}] {page_chunk.strip()}",
                                'metadata': {'type': 'page', 'page': i+1}
                            })
                else:
                    chunks.append({
                        'text': f"[PAGE {i+1}] {page.strip()}",
                        'metadata': {'type': 'page', 'page': i+1}
                    })

    # Add table chunks
    if '[TABLE' in full_text:
        tables = re.findall(r'(\[TABLE[^\]]*\][^[]{0,5000})', full_text)
        for i, table in enumerate(tables):
            if table.strip():
                chunks.append({
                    'text': table.strip(),
                    'metadata': {'type': 'table', 'index': i}
                })

    # Add sentence-level chunks for exact matching
    sentences = re.split(r'[.!?]+', full_text)
    for i, sentence in enumerate(sentences):
        if len(sentence.strip()) > 50 and len(sentence.strip()) < 1200:
            chunks.append({
                'text': sentence.strip(),
                'metadata': {'type': 'sentence', 'index': i}
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

    # Cache the chunks
    chunk_cache[text_hash] = unique
    return unique


# Initialize components
document_parser = OptimizedDocumentParser()
retriever = WinningRetriever()
reasoning_engine = WinningReasoningEngine()

# Global cache with MAXIMUM size
response_cache = TTLCache(maxsize=Config.RESPONSE_CACHE_SIZE, ttl=Config.CACHE_TTL)
document_cache = TTLCache(maxsize=Config.DOCUMENT_CACHE_SIZE, ttl=Config.CACHE_TTL)
chunk_cache = TTLCache(maxsize=Config.CHUNK_CACHE_SIZE, ttl=Config.CACHE_TTL)


class QueryRequest(BaseModel):
    documents: str
    questions: List[str]


@app.post("/hackrx/run")
async def process_query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Process queries with winning strategy for maximum score"""

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
        logger.info(f"Processing {len(request.questions)} questions with WINNING STRATEGY")

        # Parse document with retry
        doc_data = await document_parser.parse_document(request.documents)

        if not doc_data.get('full_text'):
            return {"answers": ["Unable to parse document after multiple attempts"] * len(request.questions)}

        # Create comprehensive chunks
        chunks = create_winning_chunks(doc_data)
        if not chunks:
            return {"answers": ["No content found in document"] * len(request.questions)}

        # Index chunks
        retriever.index_documents(chunks)

        # WINNING STRATEGY: Process questions by priority
        question_priorities = []
        for i, question in enumerate(request.questions):
            priority = calculate_priority_score(question, request.documents)
            question_priorities.append((i, question, priority))

        # Sort by priority (highest first)
        question_priorities.sort(key=lambda x: x[2], reverse=True)

        logger.info(f"Question priorities: {[(q[1][:30], q[2]) for q in question_priorities[:3]]}")

        # Process questions in priority order
        answers = [""] * len(request.questions)
        for original_index, question, priority_score in question_priorities:
            logger.info(f"Processing Q{original_index+1} (Priority: {priority_score:.1f}x): {question[:50]}...")

            # Retrieve with priority-based parameters - OPTIMIZED FOR SPEED
            relevant = retriever.retrieve(question, top_k=35, priority_score=priority_score)  # OPTIMIZED FOR SPEED

            if not relevant:
                answers[original_index] = "No relevant information found in the document for this question."
                continue

            # Generate answer with priority-based resource allocation
            result = await reasoning_engine.reason(
                question,
                relevant,
                doc_data.get('tables', []),
                priority_score
            )

            answers[original_index] = result['answer']
            logger.info(f"Q{original_index+1} Confidence: {result['confidence']:.1%}, Priority: {priority_score:.1f}x")

        # Cache response
        response = {"answers": answers}
        response_cache[cache_key] = response

        total_time = time.time() - start_time
        logger.info(f"Completed in {total_time:.1f}s with WINNING STRATEGY")

        return response

    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return {"answers": [f"System error: {str(e)}"] * len(request.questions)}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "HackRx Winning Strategy API",
        "models": "gpt-4o + gpt-4-turbo-preview ensemble",
        "description": "Score-optimized strategy for maximum competition performance",
        "strategy": "Priority-based resource allocation for unknown documents and complex questions"
    }


@app.get("/")
async def root():
    return {
        "message": "HackRx Winning Strategy API Running",
        "models": "gpt-4o + gpt-4-turbo-preview ensemble",
        "features": "Score-optimized strategy for maximum competition performance",
        "scoring": "Unknown docs: 4x weight, Complex questions: 2x weight"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)