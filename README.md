# News TL;DR

A powerful news summarization system built with LangGraph that automatically fetches, processes, and generates concise summaries of the latest news articles.

## Features
- Automated news article fetching using NewsAPI
- Intelligent article selection based on relevance
- Parallel processing for efficient summarization
- Bullet-point style summaries for quick reading
- Configurable search parameters and article count
- Robust error handling and retry mechanisms

## Architecture

### Core Components
- **LangGraph**: Orchestrates the workflow and manages data flow
- **GPT-4o-mini**: Handles article selection and summarization
- **NewsAPI**: Retrieves article metadata
- **BeautifulSoup**: Extracts article content
- **Asyncio**: Enables concurrent processing


## Installation

```bash
git clone <repository-url>
cd news-tldr
python -m venv venv
# On Mac/Linux
source venv/bin/activate  
# On Windows
venv\Scripts\activate

pip install -r requirements.txt
```

## Configuration
Create a .env file with the following keys:

```python
OPENAI_API_KEY=your_openai_api_key
NEWSAPI_KEY=your_newsapi_key
```

## Usage
```python
from src.core.workflow import NewsWorkflow
import asyncio

async def main():
    workflow = NewsWorkflow()
    result = await workflow.run(
        query="your search query",
        num_articles_tldr=3,
        num_searches_remaining=3
    )
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

## Project Structure

```bash
news-tldr/
├── src/
│   ├── config/         # Configuration settings
│   ├── core/           # Core workflow implementation
│   ├── tools/          # API clients and utilities
│   └── utils/          # Helper functions
├── tests/              # Test cases
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation

```

## How It Works
- **Query Processing:** Converts user queries into NewsAPI parameters
- **Article Retrieval:** Fetches relevant articles using NewsAPI
- **Content Extraction:** Scrapes full article content
- **Article Selection:** Chooses the most relevant articles
- **Parallel Summarization:** Generates concise bullet-point summaries
- **Result Formatting:** Presents summaries in a readable format

## Dependencies
langchain
langgraph
openai
newsapi-python
beautifulsoup4
aiohttp
pydantic

## Error Handling
Automatic retries for failed requests
Rate limiting for API calls
Timeout handling
Comprehensive logging

## Contributing
Fork the repository
Create a feature branch
Commit your changes
Push to the branch
Create a Pull Request

## License
### MIT License
```plaintext
You can copy and paste this into your **README.md** file, and it will be properly formatted when rendered.
```