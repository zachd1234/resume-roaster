import logging
import os
from dotenv import load_dotenv
from src.search_agent.providers.model_provider import ModelProvider
from src.search_agent.providers.search_provider import SearchProvider
from sentient_agent_framework import (
    AbstractAgent,
    DefaultServer,
    Session,
    Query,
    ResponseHandler)
from typing import AsyncIterator


load_dotenv()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SearchAgent(AbstractAgent):
    def __init__(
            self,
            name: str
    ):
        super().__init__(name)

        model_api_key = os.getenv("MODEL_API_KEY")
        if not model_api_key:
            raise ValueError("MODEL_API_KEY is not set")
        self._model_provider = ModelProvider(api_key=model_api_key)

        search_api_key = os.getenv("TAVILY_API_KEY")
        if not search_api_key:
            raise ValueError("TAVILY_API_KEY is not set") 
        self._search_provider = SearchProvider(api_key=search_api_key)


    # Implement the assist method as required by the AbstractAgent class
    async def assist(
            self,
            session: Session,
            query: Query,
            response_handler: ResponseHandler
    ):

        # Use response handler to create a text stream to stream the final 
        final_response_stream = response_handler.create_text_stream(
            "FINAL_RESPONSE"
        )

        async for chunk in self.__roast_resume(query.prompt):
            # Use the text stream to emit chunks of the final response to the client
            await final_response_stream.emit_chunk(chunk)
        
        await final_response_stream.emit_chunk("\n\nAs a favor to you, I am now giving you a life-line to save your face. Here is the updated resume:\n\n")

        async for chunk in self.__update_resume(query.prompt):
            # Use the text stream to emit chunks of the final response to the client
            await final_response_stream.emit_chunk(chunk)
        
        # Convert final response to PDF
        # Mark the text stream as complete
        await final_response_stream.complete()
        # Mark the response as complete
        await response_handler.complete()
    
    async def __roast_resume(
            self,
            resume_text: str,
    ) -> AsyncIterator[str]:
        """Process the search results."""
        roast_resume_prompt = f"Given the following resume, write a roast as if you're reviewing it in a brutal way. Don't hold back. The tone should be funny, savage, colorful, emotional, and fast-paced. Mention and joke about specific elements from the resume. Ensure it is highly personalized to the resumes unique components. Use vivid metaphors, cultural references, and exaggerations to make it more entertaining. Then finish with a 🍗 Roast-o-Meter: X/5 Sizzler: Compare the resume to a cultural moment, viral trend, or funny disaster using vivid language, playful exaggeration, and strong imagery. Limit this to one vivid, punchy sentence. Return output in a paragraph structure. Resume:{resume_text}"
        async for chunk in self._model_provider.query_stream(roast_resume_prompt):
            yield chunk


    async def __update_resume(
            self,
            resume_text: str,
    ) -> AsyncIterator[str]:
        """Process the search results."""
        with open('/Users/soham/Desktop/Sentient-Agent-Framework-Examples/examples/search_agent/src/search_agent/resume_update_prompt.txt', 'r') as f:
            update_resume_prompt = f.read() + f"\n\nResume: {resume_text}"
        async for chunk in self._model_provider.query_stream(update_resume_prompt):
            yield chunk


if __name__ == "__main__":
    # Create an instance of a SearchAgent
    agent = SearchAgent(name="Search Agent")
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    # Run the server
    server.run()