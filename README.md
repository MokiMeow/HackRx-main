# HackRx Intelligent Query-Retrieval System

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- OpenAI API Key

### Installation

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   venv\Scripts\activate  # Windows
   # source venv/bin/activate  # Linux/Mac
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API Key:**
   - Open `.env` file
   - Add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-openai-api-key-here
   ```

4. **Start the server:**
   ```bash
   uvicorn src.api.app:app --host 0.0.0.0 --port 8000
   ```

5. **API will be available at:** `http://localhost:8000`
   - Health check: `http://localhost:8000/health`
   - API docs: `http://localhost:8000/docs`

## 🤖 API Usage

### Endpoint
```
POST /hackrx/run
```

### Headers
```
Content-Type: application/json
Accept: application/json
Authorization: Bearer 0ff7ebb40ca0dcfd3650d97dcfb25b1d8a8c71496e8f5eb0b97ae7a370729c3d
```

### Request Body
```json
{
    "documents": "https://example.com/policy.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?"
    ]
}
```

### Response
```json
{
    "answers": [
        "The grace period for premium payment is thirty days...",
        "The waiting period for pre-existing diseases is 36 months..."
    ]
}
```

## 🧪 Testing in Postman
1. Set method to POST: `http://localhost:8000/hackrx/run`
2. Add the 3 headers above
3. Set body to raw JSON with your questions
4. Send request

## 📁 Project Structure

```
hackrx-solution/
├── src/api/app.py          # Main application
├── .env                    # Configuration (add your OpenAI API key here)
├── .env.example           # Environment template
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## ⚡ Features

- ✅ PDF document processing
- ✅ OpenAI-powered answer generation
- ✅ Crisp, concise responses
- ✅ Robot testing compatible
- ✅ Production ready

## 🎯 Ready for HackRx Robot Testing

System is optimized for automated testing with exact API format compliance.