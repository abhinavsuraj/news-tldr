import aiohttp
import asyncio
from bs4 import BeautifulSoup
from typing import Optional, Dict, List, Any
from urllib.parse import urlparse
import logging
from pydantic import BaseModel
from ..config.settings import Settings
import brotli
import zlib
from aiohttp import ClientResponse
import aiohttp
from asyncio import Semaphore
import aiohttp
import asyncio
import random
import time


semaphore = Semaphore(5)

class ScraperError(Exception):
    """Custom exception for scraping errors"""
    pass

class ScrapedContent(BaseModel):
    """Model for scraped article content"""
    url: str
    title: Optional[str]
    text: str
    metadata: Dict = {}




class WebScraper:
    def __init__(self):
        """Initialize the scraper with configuration"""
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        self.session: Optional[aiohttp.ClientSession] = None
        self.rate_limit_delay = 1.0  # 1 second between requests
        self.max_retries = 3
        self.timeout = aiohttp.ClientTimeout(total=30)
        
        # Common headers to mimic browser behavior
        self.headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Accept-Encoding': 'gzip, deflate',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1'
    }

    async def __aenter__(self):
        """Context manager entry with Brotli decoding support."""
        self.session = aiohttp.ClientSession(
            headers=self.headers,
            timeout=self.timeout
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if self.session:
            await self.session.close()

    async def scrape_article(self, url: str) -> Optional[Dict[str, Any]]:
        """Web scrapes to retrieve article text."""
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/85.0.4183.121 Safari/537.36'
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to fetch {url}: {response.status}")
                        return None
                    
                    # Parse the HTML content
                    text = await response.text()
                    soup = BeautifulSoup(text, 'html.parser')
                    
                    # Extract article content
                    article_text = self._extract_article_text(soup)
                    if not article_text:
                        return None
                        
                    return {
                        "url": url,
                        "content": article_text
                    }
                    
        except Exception as e:
            self.logger.error(f"Error scraping {url}: {str(e)}")
            return None


    logger = logging.getLogger(__name__)
    def fetch_article(url, max_retries=3, delay=2):
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36"
        }

        for attempt in range(max_retries):
            try:
                # Fetch the article with a timeout to avoid hanging
                response = httpx.get(url, headers=headers, timeout=10)

                # Raise exception for 4xx or 5xx status codes
                response.raise_for_status()

                # Return content if successful
                return response.text

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP Error for {url}: {e}")
                if e.response.status_code == 403:
                    logger.warning(f"403 Forbidden. Skipping {url}. Consider rechecking headers or rate limits.")
                    return None  # Stop retries for 403
                time.sleep(delay)  # Small wait before retrying

            except httpx.RequestError as e:
                logger.error(f"Request error for {url}: {e}")
                time.sleep(delay)  # Retry after delay

        logger.error(f"Max retries reached for {url}. Skipping.")
        return None  # Return None if all retries failed

    async def _fetch_content(self, url: str) -> Optional[str]:
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.190 Safari/537.36',
                    'Accept-Encoding': 'gzip, deflate'
                }
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        self.logger.error(f"Failed to fetch {url}: {response.status}")
                        return None
                        
                    try:
                        return await response.text()
                    except Exception as e:
                        self.logger.error(f"Error decoding content: {str(e)}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error fetching {url}: {str(e)}")
            return None


    async def _fetch_with_retry(self, url: str, max_retries: int = 3) -> Optional[str]:
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
            'Mozilla/5.0 (X11; Linux x86_64)'
        ]

        for attempt in range(max_retries):
            try:
                headers = {
                    'User-Agent': user_agents[attempt % len(user_agents)],
                    'Accept': 'text/html'
                }

                async with aiohttp.ClientSession() as session:
                    async with session.get(url, headers=headers, timeout=self.timeout) as response:
                        if response.status == 200:
                            return await response.text()

                        if response.status == 429:  # Rate limit
                            self.logger.warning(f"Rate limit hit for {url}. Retrying in {2 ** attempt} seconds...")
                            await asyncio.sleep(2 ** attempt)
                            continue

                        self.logger.error(f"Error {response.status} fetching {url}")
                        return None

            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} failed for {url}: {str(e)}")

        return None

    async def _extract_article_content(
        self, 
        html: str, 
        url: str
    ) -> Optional[ScrapedContent]:
        """
        Extract article content from HTML
        
        Args:
            html: Raw HTML content
            url: Source URL
            
        Returns:
            ScrapedContent object or None if extraction fails
        """
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove unwanted elements
            self._clean_html(soup)
            
            # Extract content based on domain-specific rules
            domain = urlparse(url).netloc
            content = self._extract_by_domain(soup, domain)
            
            if not content:
                content = self._extract_generic(soup)
            
            if not content:
                return None
                
            # Create scraped content object
            return ScrapedContent(
                url=url,
                title=self._extract_title(soup),
                text=content,
                metadata=self._extract_metadata(soup)
            )
            
        except Exception as e:
            self.logger.error(f"Error extracting content from {url}: {str(e)}")
            return None

    def _clean_html(self, soup: BeautifulSoup) -> None:
        """Remove unwanted HTML elements"""
        for element in soup.find_all(['script', 'style', 'nav', 'footer', 'iframe']):
            element.decompose()

    def _extract_medium(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article content specifically for Medium.com."""
        article_body = soup.find('article')
        if article_body:
            return self._clean_text(article_body.get_text())
        return None

    def _extract_bbc(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article content for BBC News."""
        article = soup.find('article')
        if article:
            return self._clean_text(article.get_text())
        return None

    def _extract_article_text(self, soup: BeautifulSoup) -> str:
        try:
            # Remove unwanted elements
            for tag in soup(['script', 'style', 'nav', 'header', 'footer']):
                tag.decompose()
                
            # Get all paragraph text
            paragraphs = soup.find_all(['p', 'article', 'div.article-body'])
            text = ' '.join(p.get_text().strip() for p in paragraphs if p.get_text().strip())
            
            # Clean up whitespace
            text = ' '.join(text.split())
            
            if not text:
                self.logger.warning("No article text found")
                return ""
                
            return text
            
        except Exception as e:
            self.logger.error(f"Error extracting article text: {str(e)}")
            return ""


    def _extract_by_domain(self, soup: BeautifulSoup, domain: str) -> Optional[str]:
        """
        Extract content using domain-specific rules
        
        Args:
            soup: BeautifulSoup object
            domain: Website domain
            
        Returns:
            Extracted text content or None
        """
        # Add domain-specific extraction rules
        domain_rules = {
            'medium.com': self._extract_medium,
            'bbc.com': self._extract_bbc,
        }
        
        extractor = domain_rules.get(domain)
        if extractor:
            return extractor(soup)
        return self._extract_generic(soup)

    def _extract_generic(self, soup: BeautifulSoup) -> Optional[str]:
        """
        Generic content extraction for unknown domains
        
        Args:
            soup: BeautifulSoup object
            
        Returns:
            Extracted text content or None
        """
        # Try common article containers
        article = soup.find('article')
        if article:
            return self._clean_text(article.get_text())
            
        # Try main content area
        main = soup.find('main')
        if main:
            return self._clean_text(main.get_text())
            
        # Try content by class names
        content_classes = ['content', 'article-content', 'post-content']
        for class_name in content_classes:
            content = soup.find(class_=class_name)
            if content:
                return self._clean_text(content.get_text())
                
        return None

    def _extract_title(self, soup: BeautifulSoup) -> Optional[str]:
        """Extract article title"""
        title = soup.find('h1')
        if title:
            return self._clean_text(title.get_text())
        return None

    def _extract_metadata(self, soup: BeautifulSoup) -> Dict:
        """Extract article metadata"""
        metadata = {}
        
        # Extract publication date
        for meta in soup.find_all('meta'):
            if 'published_time' in meta.get('property', ''):
                metadata['published_date'] = meta.get('content')
                
        # Extract author
        author = soup.find(class_=['author', 'byline'])
        if author:
            metadata['author'] = self._clean_text(author.get_text())
            
        return metadata

    def _clean_text(self, text: str) -> str:
        """Clean extracted text content"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common unwanted phrases
        unwanted = ['Advertisement', 'Cookie Policy', 'Privacy Policy']
        for phrase in unwanted:
            text = text.replace(phrase, '')
            
        return text.strip()

    async def _reset_session(self) -> None:
        """Reset aiohttp session"""
        if self.session:
            await self.session.close()
            self.session = None
