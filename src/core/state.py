from typing import TypedDict, Annotated, List

class GraphState(TypedDict):
    news_query: Annotated[str, "Input query"]
    num_searches_remaining: Annotated[int, "Search count"]
    newsapi_params: Annotated[dict, "API parameters"]
    articles_metadata: Annotated[List[dict], "Article metadata"]
    scraped_urls: Annotated[List[str], "Processed URLs"]
    potential_articles: Annotated[List[dict], "Articles to process"]
    tldr_articles: Annotated[List[dict], "Summarized articles"]
