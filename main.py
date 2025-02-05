import asyncio
import logging
from src.core.workflow import NewsWorkflow

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

async def main():
    try:
        # Initialize workflow
        logging.info("Initializing News Workflow...")
        workflow = NewsWorkflow()
        
        # Define search parameters
        search_query = "USA Stock market January 2025"
        num_articles = 3
        max_searches = 3
        
        # Execute workflow
        logging.info("Running the News Workflow...")
        result = await workflow.run(
            query=search_query,
            num_articles_tldr=num_articles,
            num_searches_remaining=max_searches
        )
        
        # Log the result for debugging
        logging.info(f"Workflow result: {result}")

        # Print results
        if result.get("success"):
            print("\nSummaries:")
            for summary in result["summaries"]:
                print(f"\nTitle: {summary['title']}")
                print(f"Summary: {summary['summary']}")
                print(f"URL: {summary['url']}")
                print("-" * 80)
        else:
            print(f"\nError: {result.get('error', 'Unknown error')}")

    except Exception as e:
        logging.error(f"Workflow execution failed: {str(e)}")

if __name__ == "__main__":
    asyncio.run(main())
