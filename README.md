# research-monitoring-agent

An adaptive AI research monitoring system that discovers, ranks, reviews, summarizes, and learns from user feedback on research papers.

## Current Project Status
- **Phase 0 (Foundation)**: Initial repository setup with minimal dependencies. No AI or DB functionality implemented yet.

## Planned High-Level Architecture
- **Language**: Python
- **Database**: PostgreSQL with pgvector for storing research data and embeddings.
- **Core Modules**: 
  - Discovery & Retrieval
  - Ranking & Filtering
  - Review & Summarization
  - Feedback Loop

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
2. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Copy `.env.example` to `.env` and configure your environment variables.

## How to Run Tests
Before running tests, ensure `src` is accessible, for example by running in the active virtual environment and setting PYTHONPATH or installing in editable mode. Since this is minimal, just run:
```bash
$env:PYTHONPATH="src"
pytest
```
