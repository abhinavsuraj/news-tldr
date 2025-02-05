import logging
from typing import Dict, List, Any
from dotenv import load_dotenv
import os
import re
from langgraph.graph import Graph, END
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from datetime import datetime, timedelta
from .state import GraphState
from ..tools.news_api import NewsAPITool
from ..tools.scraper import WebScraper
from ..tools.summarizer import ArticleSummarizer
from ..config.settings import Settings
import asyncio
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser

class NewsApiParams(BaseModel):
    """Parameters for NewsAPI query."""
    q: str = Field(description="Search query string")
    from_param: str = Field(description="Start date for article search (YYYY-MM-DD)")
    to: str = Field(description="End date for article search (YYYY-MM-DD)")
    language: str = Field(default="en", description="Article language")
    sort_by: str = Field(default="relevancy", description="Sort order for articles")
    page_size: int = Field(default=100, description="Number of articles to return")

    model_config = {
        "populate_by_name": True,
        "validate_assignment": True,
        "extra": "forbid"
    }

class NewsWorkflow:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        
        # Initialize OpenAI with API key from environment
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o-mini",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        self.news_api = NewsAPITool()
        self.scraper = WebScraper()
        self.summarizer = ArticleSummarizer()
        self.workflow = self._build_graph()


    def validate_state(self, state: GraphState) -> bool:
        """Validate the workflow state."""
        required_keys = {
            "news_query": str,
            "num_searches_remaining": int,
            "num_articles_tldr": int,
            "newsapi_params": dict,
            "articles_metadata": list,
            "scraped_urls": list,
            "potential_articles": list,
            "tldr_articles": list
        }
        
        try:
            # Check all required keys exist
            if not all(key in state for key in required_keys):
                missing_keys = [key for key in required_keys if key not in state]
                self.logger.error(f"Missing required keys: {missing_keys}")
                return False
                
            # Validate types
            for key, expected_type in required_keys.items():
                if not isinstance(state[key], expected_type):
                    self.logger.error(f"Invalid type for {key}: expected {expected_type}, got {type(state[key])}")
                    return False
                    
            return True
            
        except Exception as e:
            self.logger.error(f"State validation failed: {str(e)}")
            return False


    def select_top_urls(self, state: GraphState) -> GraphState:
        """Select top articles based on relevance to query."""
        try:
            self.logger.info("Starting URL selection process")
            news_query = state["news_query"]
            num_articles_tldr = state["num_articles_tldr"]
            potential_articles = state["potential_articles"]
            
            if not potential_articles:
                self.logger.warning("No potential articles available for selection")
                return state
                
            # Format article metadata for LLM
            formatted_metadata = "\n".join([
                f"URL: {article['url']}\nTitle: {article.get('title', '')}\nDescription: {article.get('description', '')}\n"
                for article in potential_articles
            ])
            
            prompt = f"""
            Based on the user news query: "{news_query}"
            
            Select up to {num_articles_tldr} most relevant articles from the following list.
            Only include URLs that are directly relevant to the query.
            
            Available articles:
            {formatted_metadata}
            
            Respond with only the URLs of selected articles, one per line.
            """
            
            prompt = f"""
            Based on the user news query: "{news_query}"

            Select up to {num_articles_tldr} most relevant articles about {news_query}.
            For each selected article, output ONLY its URL on a new line.

            Available articles:
            {formatted_metadata}

            Output format example:
            https://example.com/article1
            https://example.com/article2
            """
            self.logger.info(f"Sending prompt to LLM for URL selection")
            result = self.llm.invoke(prompt).content
            
            # Extract urls using regex
            url_pattern = r'(https?://[^\s",]+)'
            urls = re.findall(url_pattern, result)
            
            if not urls:
                self.logger.warning("No URLs extracted from LLM response")
                # Use first 3 articles as fallback
                urls = [article['url'] for article in potential_articles[:3]]
                self.logger.info(f"Using fallback: selected first {len(urls)} articles")
                return state
                
            self.logger.info(f"Found {len(urls)} relevant URLs")
            
            # Filter articles based on selected urls
            tldr_articles = [
                article for article in potential_articles 
                if article['url'] in urls
            ]
            
            if not tldr_articles:
                self.logger.warning("No matching articles found after URL filtering")
                return state
                
            self.logger.info(f"Selected {len(tldr_articles)} articles for summarization")
            state["tldr_articles"] = tldr_articles
            return state
            
        except Exception as e:
            self.logger.error(f"Error in select_top_urls: {str(e)}")
            state["error"] = f"URL selection failed: {str(e)}"
            return state


    def articles_text_decision(self, state: GraphState) -> str:
        """Check results of retrieve_articles_text to determine next step."""
        if state["num_searches_remaining"] == 0:
            # if no articles with text were found return END
            if len(state["potential_articles"]) == 0:
                state["formatted_results"] = "No articles with text found."
                return "END"
            # if some articles were found, move on to selecting the top urls
            else:
                return "select_top_urls"
        else:
            # if the number of articles found is less than the number of articles to summarize
            if len(state["potential_articles"]) < state["num_articles_tldr"]:
                return "generate_newsapi_params"
            # otherwise move on to selecting the top urls
            else:
                return "select_top_urls"

    def format_results(self, state: GraphState) -> GraphState:
        """Format the results for display."""
        try:
            # Format search terms used
            q = [params["q"] for params in state.get("past_searches", [])]
            formatted_results = f"Search terms used: {', '.join(q)}\n\n"
            
            # Format article summaries
            if state.get("tldr_articles"):
                summaries = []
                for article in state["tldr_articles"]:
                    if article.get("summary"):
                        summaries.append(article["summary"])
                formatted_results += "\n\n".join(summaries)
            else:
                formatted_results = "No articles found."
                
            state["formatted_results"] = formatted_results
            return state
            
        except Exception as e:
            self.logger.error(f"Error formatting results: {str(e)}")
            state["error"] = f"Failed to format results: {str(e)}"
            return state

    def _build_graph(self) -> Graph:
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("generate_newsapi_params", self.generate_newsapi_params)
        workflow.add_node("fetch_articles", self.retrieve_articles_metadata)
        workflow.add_node("process_articles", self.process_articles)
        workflow.add_node("select_top_urls", self.select_top_urls)
        workflow.add_node("summarize", self.generate_summaries)
        workflow.add_node("format_results", self.format_results)
        
        # Set entry point
        workflow.set_entry_point("generate_newsapi_params")
        
        # Add edges
        workflow.add_edge("generate_newsapi_params", "fetch_articles")
        workflow.add_edge("fetch_articles", "process_articles")
        
        # Add conditional edges
        workflow.add_conditional_edges(
            "process_articles",
            self.articles_text_decision,
            {
                "generate_newsapi_params": "generate_newsapi_params",
                "select_top_urls": "select_top_urls",
                "END": END
            }
        )
        
        workflow.add_edge("select_top_urls", "summarize")
        workflow.add_edge("summarize", "format_results")
        workflow.add_edge("format_results", END)
        
        return workflow.compile()

    async def generate_newsapi_params(self, state: GraphState) -> GraphState:
        """Generate parameters for NewsAPI query."""
        try:
            today = datetime.now()
            thirty_days_ago = (today - timedelta(days=29)).strftime("%Y-%m-%d")
            today_date = today.strftime("%Y-%m-%d")
            
            # Create prompt template
            template = """
            Create a param dict for the News API based on the user query:
            {query}
            
            CRITICAL DATE CONSTRAINTS:
            - 'from_param' MUST be {thirty_days_ago} or later
            - 'to' MUST be {today_date}
            
            Past searches: {past_searches}
            Searches remaining: {num_searches_remaining}
            
            Return ONLY a JSON object with these fields (no markdown, no code blocks):
            - q: string (search query terms)
            - from_param: string (date in YYYY-MM-DD format)
            - to: string (date in YYYY-MM-DD format)
            - language: string (default "en")
            - sort_by: string (default "relevancy")
            - page_size: integer (default 100)
            """
            
            prompt = ChatPromptTemplate.from_template(template)
            
            # Format prompt
            formatted_prompt = prompt.format(
                query=state["news_query"],
                thirty_days_ago=thirty_days_ago,
                today_date=today_date,
                past_searches=state.get("past_searches", []),
                num_searches_remaining=state["num_searches_remaining"]
            )
            
            # Get LLM response
            response = await self.llm.ainvoke(formatted_prompt)
            content = response.content
            
            # Clean response content
            if "```" in content:  # Check if content contains code blocks
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            
            # Parse response into NewsApiParams
            params = NewsApiParams.model_validate_json(content.strip())
            
            # Update state
            state["newsapi_params"] = params.model_dump()
            if "past_searches" not in state:
                state["past_searches"] = []
            state["past_searches"].append(params.model_dump())
            
            return state
            
        except Exception as e:
            self.logger.error(f"Error generating NewsAPI parameters: {str(e)}")
            state["error"] = f"Failed to generate NewsAPI parameters: {str(e)}"
            return state




    async def fetch_articles(self, params: Dict[str, Any]) -> List[Dict[str, Any]]:
        try:
            if not self.settings.NEWSAPI_KEY:
                self.logger.error("NewsAPI key not found")
                return []
                
            response = await self._make_api_request(params)
            if not response or 'articles' not in response:
                self.logger.error(f"Invalid NewsAPI response: {response}")
                return []
                
            return response['articles']
            
        except Exception as e:
            self.logger.error(f"NewsAPI fetch failed: {str(e)}")
            return []




    async def retrieve_articles_metadata(self, state: GraphState) -> GraphState:
        try:
            if not state["newsapi_params"]:
                self.logger.error("NewsAPI parameters are missing.")
                state["error"] = "NewsAPI parameters are not provided."
                return state

            self.logger.info(f"Fetching articles using parameters: {state['newsapi_params']}")
            
            articles = await self.news_api.fetch_articles(state["newsapi_params"])
            if not articles:
                self.logger.warning("No articles returned from NewsAPI.")
                state["error"] = "No articles fetched from NewsAPI."
                return state

            self.logger.info(f"Fetched {len(articles)} articles.")
            state["articles_metadata"] = articles  # Store articles in the state
            return state

        except Exception as e:
            self.logger.error(f"Error fetching articles: {str(e)}")
            state["error"] = f"Failed to fetch articles: {str(e)}"
            return state


    async def process_articles(self, state: GraphState) -> GraphState:
        """Process articles from metadata to full text."""
        try:
            articles_to_process = state["articles_metadata"][:self.settings.BATCH_SIZE]
            processed_articles = []
            
            for article in articles_to_process:
                if article.get("url") not in state["scraped_urls"]:
                    try:
                        content = await self.scraper.scrape_article(article["url"])
                        if content:
                            processed_articles.append({
                                "title": article.get("title", ""),
                                "url": article["url"],
                                "content": content["content"]
                            })
                            state["scraped_urls"].append(article["url"])
                    except Exception as e:
                        self.logger.error(f"Error processing article {article['url']}: {str(e)}")
                        continue
                        
            state["potential_articles"].extend(processed_articles)
            return state
            
        except Exception as e:
            self.logger.error(f"Error in process_articles: {str(e)}")
            return {**state, "error": str(e)}

    async def generate_summaries(self, state: GraphState) -> GraphState:
        """Summarize articles in parallel using asyncio."""
        try:
            if not state.get("tldr_articles"):
                self.logger.warning("No articles to summarize")
                return state
                
            self.logger.info(f"Starting summarization for {len(state['tldr_articles'])} articles")
            
            # Create tasks for parallel summarization with timeout
            async def summarize_with_timeout(article):
                try:
                    formatted_prompt = f"""
                    Create a * bulleted summarizing tldr for the article:
                    {article.get('content', '')}
                    Be sure to follow the following format exactly with nothing else:
                    {article.get('title', '')}
                    {article.get('url', '')}
                    * tl;dr bulleted summary
                    * use bullet points for each sentence
                    """
                    return await asyncio.wait_for(
                        self.llm.ainvoke(formatted_prompt),
                        timeout=30
                    )
                except asyncio.TimeoutError:
                    self.logger.error(f"Summarization timed out for {article.get('url')}")
                    return None
                    
            tasks = [summarize_with_timeout(article) for article in state["tldr_articles"]]
            
            # Execute summaries in parallel with overall timeout
            try:
                summaries = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=120
                )
                
                # Update articles with successful summaries
                for i, summary in enumerate(summaries):
                    if summary and not isinstance(summary, Exception):
                        state["tldr_articles"][i]["summary"] = summary.content
                        
                self.logger.info(f"Successfully generated {len([s for s in summaries if s])} summaries")
                return state
                
            except asyncio.TimeoutError:
                self.logger.error("Overall summarization timed out")
                state["error"] = "Summarization timed out"
                return state
                
        except Exception as e:
            self.logger.error(f"Error in generate_summaries: {str(e)}")
            state["error"] = f"Failed to generate summaries: {str(e)}"
            return state

    async def run(
        self, 
        query: str,
        num_articles_tldr: int = 3,
        num_searches_remaining: int = 3
    ) -> Dict[str, Any]:
        try:
            self.logger.info(f"Starting workflow with query: {query}")
            
            initial_state = {
                "news_query": query,
                "num_searches_remaining": num_searches_remaining,
                "num_articles_tldr": num_articles_tldr,
                "newsapi_params": {},
                "articles_metadata": [],
                "scraped_urls": [],
                "potential_articles": [],
                "tldr_articles": [],
                "formatted_results": ""
            }
            
            if not self.validate_state(initial_state):
                return {
                    "success": False,
                    "summaries": [],
                    "error": "Invalid initial state"
                }
                
            result = await asyncio.wait_for(
                self.workflow.ainvoke(initial_state),
                timeout=self.settings.WORKFLOW_TIMEOUT
            )
            
            if result.get("error"):
                return {
                    "success": False,
                    "summaries": [],
                    "error": result["error"]
                }
                
            return {
                "success": True,
                "summaries": [
                    {
                        "title": article.get("title", ""),
                        "url": article.get("url", ""),
                        "summary": article.get("summary", "")
                    }
                    for article in result.get("tldr_articles", [])
                ],
                "error": None
            }
            
        except asyncio.TimeoutError:
            self.logger.error("Workflow execution timed out")
            return {
                "success": False,
                "summaries": [],
                "error": "Operation timed out"
            }
        except Exception as e:
            self.logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "success": False,
                "summaries": [],
                "error": f"Workflow execution failed: {str(e)}"
            }