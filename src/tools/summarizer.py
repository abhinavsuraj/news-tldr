from typing import Optional, List, Dict
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import asyncio
import logging
from ..config.settings import Settings

class SummaryRequest(BaseModel):
    """Model for summarization request parameters"""
    text: str
    max_length: int = Field(default=150, le=500)
    style: str = Field(
        default="concise",
        pattern="^(concise|detailed|bullet_points)$"
    )
    focus_points: Optional[List[str]] = None

class SummaryResponse(BaseModel):
    """Model for summarization response"""
    summary: str
    key_points: List[str]
    metadata: Dict = {}

class SummarizerError(Exception):
    """Custom exception for summarization errors"""
    pass

class ArticleSummarizer:
    def __init__(self):
        """Initialize the summarizer with configuration"""
        self.settings = Settings()
        self.logger = logging.getLogger(__name__)
        self.llm = ChatOpenAI(
            model_name=self.settings.MODEL_NAME,
            temperature=0.3,
            max_tokens=500
        )
        self.max_retries = 3
        self.chunk_size = 4000  # Maximum tokens per chunk

    async def summarize(
        self,
        text: str,
        style: str = "concise",
        max_length: int = 150
    ) -> SummaryResponse:
        """
        Generate article summary with retry logic
        
        Args:
            text: Article text to summarize
            style: Summarization style (concise/detailed/bullet_points)
            max_length: Maximum summary length
            
        Returns:
            SummaryResponse object
            
        Raises:
            SummarizerError: If summarization fails after retries
        """
        request = SummaryRequest(
            text=text,
            style=style,
            max_length=max_length
        )

        for attempt in range(self.max_retries):
            try:
                # Split long text into chunks
                chunks = self._split_text(request.text)
                
                # Summarize each chunk
                chunk_summaries = await self._process_chunks(chunks, request)
                
                # Combine chunk summaries
                if len(chunk_summaries) > 1:
                    final_summary = await self._combine_summaries(
                        chunk_summaries,
                        request
                    )
                else:
                    final_summary = chunk_summaries[0]
                
                # Extract key points
                key_points = await self._extract_key_points(final_summary)
                
                return SummaryResponse(
                    summary=final_summary,
                    key_points=key_points,
                    metadata={"style": style, "chunks": len(chunks)}
                )

            except Exception as e:
                self.logger.error(f"Summarization attempt {attempt + 1} failed: {str(e)}")
                if attempt == self.max_retries - 1:
                    raise SummarizerError(f"Failed to generate summary: {str(e)}")
                await asyncio.sleep(2 ** attempt)  # Exponential backoff

    def _split_text(self, text: str) -> List[str]:
        """
        Split long text into manageable chunks
        
        Args:
            text: Input text
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0

        for word in words:
            word_length = len(word)
            if current_length + word_length > self.chunk_size:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = word_length
            else:
                current_chunk.append(word)
                current_length += word_length

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    async def _process_chunks(
        self,
        chunks: List[str],
        request: SummaryRequest
    ) -> List[str]:
        """
        Process text chunks in parallel
        
        Args:
            chunks: List of text chunks
            request: Summarization parameters
            
        Returns:
            List of chunk summaries
        """
        tasks = []
        for chunk in chunks:
            tasks.append(self._summarize_chunk(chunk, request))
        return await asyncio.gather(*tasks)

    async def _summarize_chunk(
        self,
        text: str,
        request: SummaryRequest
    ) -> str:
        """
        Summarize individual text chunk
        
        Args:
            text: Text chunk
            request: Summarization parameters
            
        Returns:
            Chunk summary
        """
        style_prompts = {
            "concise": "Create a brief, focused summary",
            "detailed": "Create a comprehensive summary",
            "bullet_points": "Create a summary in bullet points"
        }

        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            You are an expert summarizer. Your task is to create clear,
            accurate summaries while preserving key information.
            Style: {style_prompt}
            Maximum length: {max_length} words
            """),
            ("user", "{text}")
        ])

        messages = prompt.format_messages(
            style_prompt=style_prompts[request.style],
            max_length=request.max_length,
            text=text
        )

        response = await self.llm.ainvoke(messages)
        return response.content

    async def _combine_summaries(
        self,
        summaries: List[str],
        request: SummaryRequest
    ) -> str:
        """
        Combine multiple chunk summaries
        
        Args:
            summaries: List of chunk summaries
            request: Summarization parameters
            
        Returns:
            Combined summary
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Combine these summaries into a coherent {style} summary.
            Maximum length: {max_length} words.
            Maintain consistency and avoid redundancy.
            """),
            ("user", "{summaries}")
        ])

        messages = prompt.format_messages(
            style=request.style,
            max_length=request.max_length,
            summaries="\n".join(summaries)
        )

        response = await self.llm.ainvoke(messages)
        return response.content

    async def _extract_key_points(self, summary: str) -> List[str]:
        """
        Extract key points from summary
        
        Args:
            summary: Generated summary
            
        Returns:
            List of key points
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", """
            Extract 3-5 key points from this summary.
            Each point should be concise and informative.
            Return only the points, one per line.
            """),
            ("user", "{summary}")
        ])

        messages = prompt.format_messages(summary=summary)
        response = await self.llm.ainvoke(messages)
        
        # Process response into list of points
        points = [
            point.strip().lstrip('•-*')
            for point in response.content.split('\n')
            if point.strip()
        ]
        
        return points

    async def summarize_batch(
        self,
        texts: List[str],
        style: str = "concise",
        max_length: int = 150
    ) -> List[SummaryResponse]:
        """
        Summarize multiple texts in parallel
        
        Args:
            texts: List of texts to summarize
            style: Summarization style
            max_length: Maximum summary length
            
        Returns:
            List of SummaryResponse objects
        """
        tasks = []
        for text in texts:
            tasks.append(self.summarize(text, style, max_length))
        return await asyncio.gather(*tasks)
