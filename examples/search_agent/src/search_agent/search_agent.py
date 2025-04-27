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
            roast_text: str = "",
    ) -> AsyncIterator[str]:
        """Process the search results."""
        # with open('resume_update_prompt.txt', 'r') as f:
        #     update_resume_prompt = f.read() + f"\n\nResume: {resume_text}"
        update_resume_prompt = f"""
        ROLE & EXPERTISE
You are an ATS-optimization specialist and technical résumé writer specializing in Computer Science and Machine Learning roles. You have deep knowledge of relevant technologies and industry best practices. You are also skilled at distilling constructive feedback from AI-generated critiques.

GOAL
Craft a single-page résumé in plain text only that is laser-focused on the attached Job Description (if provided) or inferred from the resume text. The result must:

1. Accept OCR-extracted resume text as input.
2. Accept AI-generated critique (Roast Text) as input and incorporate the feedback.
3. Infer the job role from the resume text if a Job Description is not provided.
4. Maximize ATS visibility – weave in the job’s exact keywords (identified from the Job Description or inferred from the resume).
5. Show measurable impact – every bullet follows the STAR pattern and starts with a strong verb, contains a metric, and ends with a concrete result.
6. Stay concise & technical – no soft-skill fluff unless the description demands it.
7. Fit on one US-letter page – adjust spacing or trim less-relevant details automatically.
8. Generate the new resume in plain text format.

INPUT
Roast Text: {roast_text}
Resume Text + Job Description (Optional): {resume_text}

PROCESS THINKING
Think step-by-step:
1. Parse the job keywords & required tech from the Job Description (if provided) or infer from the resume text.
2. Distill constructive feedback from the Roast Text.
3. Map keywords and feedback to candidate achievements.
4. Rewrite bullets in STAR, incorporating feedback.
5. Generate data in the plain text format.
6. Re-scan for missing keywords or metrics; revise until optimized.

DELIVERABLE
Immediately output the finalized plain text source as plain text only.

OUTPUT
Return ONLY the completed plain text source – no commentary or code fences.
Only include sections that improve ATS ranking (omit “Projects” or “Additional Information” if they do not strengthen alignment).
Preface the “Skills” list with the exact header Technical • Languages • Tools to match common ATS parsers.
BEGIN NOW – remember: deliver just the plain text code, nothing else.
        """
        async for chunk in self._model_provider.query_stream(update_resume_prompt):
            yield chunk


if __name__ == "__main__":
    # Create an instance of a SearchAgent
    agent = SearchAgent(name="Search Agent")
    # Create a server to handle requests to the agent
    server = DefaultServer(agent)
    # Run the server
    server.run()
