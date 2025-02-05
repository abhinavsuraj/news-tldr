import pytest
from src.core.workflow import NewsWorkflow

@pytest.mark.asyncio
async def test_news_workflow():
    workflow = NewsWorkflow()
    result = await workflow.run("test query")
    assert result is not None
