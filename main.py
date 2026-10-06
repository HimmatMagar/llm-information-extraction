import json
import os

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint


load_dotenv()

class ExtractedInfo(BaseModel):
    name: str | None
    job_title: str | None
    company: str | None
    location: str | None
    years_experience: int | None = Field(default=None, ge=0)
    email: str | None


llm = HuggingFaceEndpoint(
    repo_id="openai/gpt-oss-120b",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
    temperature=0,
    max_new_tokens=512,
)

model = ChatHuggingFace(llm=llm)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an information extraction system.

        Extract information from the provided text.

        Return ONLY valid JSON.

        The JSON must contain exactly these fields:

        {{
            "name": string or null,
            "job_title": string or null,
            "company": string or null,
            "location": string or null,
            "years_experience": integer or null,
            "email": string or null
        }}

        Rules:
        - Do not invent information.
        - Use null when information is missing.
        - years_experience must be an integer.
        - Return JSON only.
        """
    ),
    (
        "human",
        "{text}"
    )
])

chain = prompt | model

def extract_information(text: str) -> ExtractedInfo:

    response = chain.invoke({
        "text": text
    })
    data = json.loads(response.content)

    return ExtractedInfo.model_validate(data)


if __name__ == "__main__":

    text = """
    John Smith works as a Machine Learning Engineer
    at Google in California.

    He has 3 years of experience.

    His email is john@example.com.
    """

    result = extract_information(text)
    print(result.model_dump_json(indent=2))