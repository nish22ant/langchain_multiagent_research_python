from src.agents.agents import build_search_agent, build_scrape_agent, writer_chain, critic_chain

def run_research_pipeline(topic: str) -> dict:
    state = {}

    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)

    # search agent
    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages":[
            ("user", f"Find the recent and reliable information on {topic}")
        ]
    })

    state["search_result"] = search_result['messages'][-1].content
    print("\n search result ",state['search_result'])


    # scape agent
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)

    scrape_agent = build_scrape_agent()
    scrape_result = scrape_agent.invoke(
        {
            "messages": [("user",
                          f"Based on the following search results about '{topic}', "
                          f"pick the most relevant URL and scrape it for deeper content.\n\n"
                          f"Search Results:\n{state['search_result'][:800]}"
                          )]
        },
        config={"recursion_limit": 8},
    )

    state["scrape_result"] = scrape_result["messages"][-1].content
    print("\n scrape result ", state['scrape_result'])

    # writer chain
    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_result']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scrape_result']}"
    )

    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": research_combined
    })

    print("\n Final Report\n",state['report'])

    # critic chain
    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)

    state["feedback"] = critic_chain.invoke({
        "report": state["report"]
    })

    print("\n critic report \n", state['feedback'])

    return state



