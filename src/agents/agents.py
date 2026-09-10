# import libraries
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools.tools import web_search, scrape_url
from dotenv import load_dotenv


# load env variable
load_dotenv()

# model initialization
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# create search agent
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt="You are a professional web searcher"
    )

# create web scrape agent
def build_scrape_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a professional web scraper. Given search results, pick the "
            "single most relevant URL and call scrape_url on it exactly once, then "
            "summarize what you found. Do not call tools more than twice."
        ),
    )

# create ChatPromptTemplate for writer
writer_prompt = ChatPromptTemplate.from_messages([("system", "You are expert research writer. Write a clear, structured and well thought report of a topic"),
    ("user", 

    """Write a detailed research report on the topic below.

    Topic: {topic}

    Research Gathered: {research}

    Structure the report as:
    - Introduction
    - Key Findings (minimum 3 well-explained points)
    - Conclusion
    - Sources (list all URLs found in the research)

    Be detailed, factual and professional.""")
])

# writer chain
writer_chain = writer_prompt | llm | StrOutputParser()


# critic prompt
critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a sharp and constructive research critic. Be honest and specific."),
    ("user", 
    """Review the research report below and evaluate it strictly.

    Report: {report}

    Respond in this exact format:

    Score: X/10

    Strengths:
    - ...
    - ...

    Areas to Improve:
    - ...
    - ...

    One line verdict:
    ...""")
])

# critic chain
critic_chain = critic_prompt | llm | StrOutputParser()