from typing import Dict, List, Any, Optional
from newsapi import NewsApiClient
import asyncio
from newsapi import NewsApiClient
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pydantic import BaseModel, Field
import httpx
import logging
import os
from ..config.settings import Settings
from ..utils.helpers import calculate_date_range

def __init__(self):
    load_dotenv()
    self.settings = Settings()
    if not self.settings.NEWSAPI_KEY:
        raise ValueError("NewsAPI key not found")
    self.client = NewsApiClient(api_key=self.settings.NEWSAPI_KEY)
    self.logger = logging.getLogger(__name__)

class NewsAPIResponse(BaseModel):
    """Pydantic model for NewsAPI response validation"""
    status: str
    totalResults: int
    articles: List[Dict]

class NewsAPIError(Exception):
    """Custom exception for NewsAPI related errors"""
    pass

class NewsArticle(BaseModel):
    """Model for validated article data"""
    title: str
    url: str
    source: Dict = Field(..., description="Source information")
    publishedAt: str
    description: Optional[str] = None
    content: Optional[str] = None

class NewsAPITool:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.api_key = os.getenv("NEWSAPI_KEY")
        self.base_url = "https://newsapi.org/v2/everything"

    async def fetch_articles(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            if not self.api_key:
                self.logger.error("NewsAPI key is not configured.")
                return []

            async with httpx.AsyncClient() as client:
                response = await client.get(
                    self.base_url,
                    params={**params, "apiKey": self.api_key}
                )
                response.raise_for_status()  # Raise an exception for HTTP errors
                data = response.json()

                # Check for API errors in the response
                if data.get("status") != "ok":
                    self.logger.error(f"NewsAPI Error: {data.get('message', 'Unknown error')}")
                    return []

                return data.get("articles", [])

        except Exception as e:
            self.logger.error(f"Failed to fetch articles from NewsAPI: {str(e)}")
            return []


    def _validate_date_params(self, params: Dict) -> Dict:
        """
        Validate and adjust date parameters to ensure they're within allowed range
        
        Args:
            params: Original parameters dictionary
            
        Returns:
            Updated parameters dictionary with valid dates
        """
        thirty_days_ago, today = calculate_date_range()
        
        # Ensure from_param is not more than 30 days ago
        if 'from_param' in params:
            from_date = datetime.strptime(params['from_param'], '%Y-%m-%d')
            if from_date < datetime.strptime(thirty_days_ago, '%Y-%m-%d'):
                params['from_param'] = thirty_days_ago
                self.logger.warning("Adjusted from_param to 30 days ago limit")

        # Ensure to date is not in the future
        if 'to' in params:
            to_date = datetime.strptime(params['to'], '%Y-%m-%d')
            if to_date > datetime.strptime(today, '%Y-%m-%d'):
                params['to'] = today
                self.logger.warning("Adjusted to param to today's date")

        return params

    async def _make_api_request(self, params: Dict) -> Dict:
        """
        Make the actual API request with rate limiting
        
        Args:
            params: Validated parameters dictionary
            
        Returns:
            Raw API response
            
        Raises:
            NewsAPIError: If API request fails
        """
        try:
            # Implement rate limiting
            await asyncio.sleep(self.rate_limit_delay)
            
            # Make synchronous API call in a thread pool
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.get_everything,
                **params
            )
            
            return response

        except Exception as e:
            raise NewsAPIError(f"API request failed: {str(e)}")

    async def _process_articles(self, articles: List[Dict]) -> List[Dict]:
        """
        Process and validate individual articles
        
        Args:
            articles: List of raw article dictionaries
            
        Returns:
            List of validated and processed articles
        """
        processed_articles = []
        
        for article in articles:
            try:
                # Validate article structure
                validated_article = NewsArticle(**article)
                
                # Additional filtering/processing if needed
                if self._is_valid_article(validated_article):
                    processed_articles.append(validated_article.dict())
                    
            except Exception as e:
                self.logger.warning(f"Skipping invalid article: {str(e)}")
                continue
                
        return processed_articles

    def _is_valid_article(self, article: NewsArticle) -> bool:
        """
        Additional validation checks for articles
        
        Args:
            article: Validated article object
            
        Returns:
            Boolean indicating if article meets all criteria
        """
        # Skip articles without meaningful content
        if not article.description and not article.content:
            return False
            
        # Skip articles with very short titles
        if len(article.title) < 10:
            return False
            
        # Add any other validation rules as needed
        
        return True

    async def get_sources(self, language: str = 'en') -> List[Dict]:
        """
        Get available news sources
        
        Args:
            language: Language code for filtering sources
            
        Returns:
            List of available news sources
        """
        try:
            sources = await asyncio.get_event_loop().run_in_executor(
                None,
                self.client.get_sources,
                {'language': language}
            )
            return sources['sources']
            
        except Exception as e:
            raise NewsAPIError(f"Failed to fetch sources: {str(e)}")
