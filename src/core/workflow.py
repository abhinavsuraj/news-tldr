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
            
            self.logger.info(f"Sending prompt to LLM for URL selection")
            result = self.llm.invoke(prompt).content
            
            # Extract urls using regex
            url_pattern = r'(https?://[^\s",]+)'
            urls = re.findall(url_pattern, result)
            
            if not urls:
                self.logger.warning("No URLs extracted from LLM response")
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

    def _build_graph(self) -> Graph:
        workflow = Graph()
        
        # Add nodes
        workflow.add_node("generate_newsapi_params", self.generate_newsapi_params)
        workflow.add_node("fetch_articles", self.retrieve_articles_metadata)
        workflow.add_node("process_articles", self.process_articles)
        workflow.add_node("select_top_urls", self.select_top_urls)
        workflow.add_node("summarize", self.generate_summaries)
        
        # Set entry point
        workflow.set_entry_point("generate_newsapi_params")
        
        # Add edges
        workflow.add_edge("generate_newsapi_params", "fetch_articles")
        workflow.add_edge("fetch_articles", "process_articles")
        
        # Add conditional edges for process_articles
        workflow.add_conditional_edges(
            "process_articles",
            self.articles_text_decision,
            {
                "generate_newsapi_params": "generate_newsapi_params",
                "select_top_urls": "select_top_urls",
                "END": END
            }
        )
        
        # Add remaining edges
        workflow.add_edge("select_top_urls", "summarize")
        workflow.add_edge("summarize", END)
        
        return workflow.compile()


    async def generate_newsapi_params(self, state: GraphState) -> GraphState:
        """Generates NewsAPI parameters based on the query and date constraints."""
        
        today = datetime.now()
        thirty_days_ago = (today - timedelta(days=29)).strftime("%Y-%m-%d")
        today_date = today.strftime("%Y-%m-%d")

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Generate NewsAPI parameters based on the query."),
            ("user", """
            Query: {query}
            Date range: {start_date} to {end_date}
            Remaining searches: {searches}
            """)
        ])

        params = await self.llm.ainvoke(
            prompt.format_messages(
                query=state["news_query"],
                start_date=thirty_days_ago,
                end_date=today_date,
                searches=state["num_searches_remaining"]
            )
        )

        # Validate and sanitize parameters
        newsapi_params = {
            "q": state["news_query"],
            "from_param": thirty_days_ago,
            "to": today_date,
            "language": "en",
            "sort_by": "relevancy",
            "page_size": 100
        }

        state["newsapi_params"] = newsapi_params
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
        """Summarize the articles based on full text."""
        try:
            summaries = []
            for article in state["tldr_articles"]:
                try:
                    summary = await self.summarizer.summarize(article["content"])
                    if summary:
                        summaries.append({
                            "url": article["url"],
                            "title": article.get("title", ""),
                            "summary": summary
                        })
                except Exception as e:
                    self.logger.error(f"Error summarizing article {article.get('url')}: {str(e)}")
                    continue
            
            if not summaries:
                self.logger.warning("No summaries generated")
                state["error"] = "Failed to generate any summaries"
                return state
                
            state["summaries"] = summaries
            return state
            
        except Exception as e:
            self.logger.error(f"Error in generate_summaries: {str(e)}")
            state["error"] = f"Failed to generate summaries: {str(e)}"
            return state


    async def check_continue(self, state: GraphState) -> bool:
        """Determines if the workflow should continue."""
        return (
            state["num_searches_remaining"] > 0 
            and len(state["articles_metadata"]) > 0 
            and "error" not in state
        )

    async def run(
        self, 
        query: str,
        num_articles_tldr: int = 3,
        num_searches_remaining: int = 3
    ) -> Dict[str, Any]:
        """Execute the news workflow."""
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
                "tldr_articles": []
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
                "summaries": result.get("summaries", []),
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
