import asyncio
from datetime import datetime
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, MessagesState, StateGraph
from langgraph.checkpoint.memory import MemorySaver

# Import all tools
from backend.agents.tools.research_tools import (
    arxiv_tool, 
    web_tool,     
)
from backend.agents.tools.rag_tools import (
    rag_search,
    rag_filtered_search
)
from backend.agents.models import models

class AgentState(MessagesState, total=False):
    """Enhanced state tracking with visual elements"""
    tool_outputs: Dict[str, Any]
    rag_results: Dict[str, List[Dict[str, str]]]
    visual_elements: List[Dict[str, str]]
    final_report: str | None
    report_metadata: Dict[str, Any]


def extract_visual_elements(rag_results: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Extract visual elements from RAG results

    Args:
        rag_results: Dictionary containing lists of visual elements
                    Expected format: {'picture': [{'content': str, 'source': str,
                    'content_type': str, 'score': float, 'aws_link': str}, ...]}

    Returns:
        List of dictionaries containing processed visual elements
    """
    visuals = []
    visual_types = ['table_image', 'picture', 'chart', 'graph']

    # Iterate through each type of visual content in rag_results
    for content_type, results in rag_results.items():
        # Verify results is a list
        if not isinstance(results, list):
            continue

        # Process each result in the list
        for result in results:
            if result.get('content_type') in visual_types:
                visual = {
                    'type': result.get('content_type'),
                    'url': result.get('aws_link', ''),  # Changed from image_path to aws_link
                    'description': result.get('content', ''),  # Changed from description to content
                    'source': result.get('source', ''),
                    'score': result.get('score', 0.0)  # Added score field
                }
                visuals.append(visual)

    return visuals
async def research_with_tools(state: AgentState, config: RunnableConfig) -> AgentState:
    """Enhanced research process with visual element handling"""
    print("\n====== Starting Comprehensive Research ======")
    
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    query = state["messages"][-1].content
    
    research_prompt = """As a research analyst, gather comprehensive information using all available tools:

    1. Use web_search for current information and recent developments
    2. Use fetch_arxiv for academic research and technical details
    3. Use rag_search for relevant internal knowledge
    4. Use rag_filtered_search specifically for:
       - Visual content (charts, graphs, images)
       - Tabular data
       - Technical specifications
       
    Prioritize finding:
    - Visual elements that support key points
    - Quantitative data and statistics
    - Technical details and specifications
    - Recent developments and trends

    Structure your research systematically and ensure you gather visual elements."""
    
    messages = [
        SystemMessage(content=research_prompt),
        HumanMessage(content=query)
    ]
    
    # Bind all tools to model
    model_with_tools = m.bind_tools([
        web_tool, 
        arxiv_tool, 
        rag_search, 
        rag_filtered_search
    ])
    
    print("\n====== Executing Research Plan ======")
    response = await model_with_tools.ainvoke(messages, config)
    
    # Initialize results containers
    tool_outputs = {}
    rag_results = {}
    
    if response.tool_calls:
        print("\n====== Processing Tool Calls ======")
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"\nExecuting {tool_name} with args: {tool_args}")
            
            try:
                if tool_name == "ArXiv":
                    result = await arxiv_tool.ainvoke(tool_args)
                    tool_outputs["arxiv"] = result
                elif tool_name == "WebSearch":
                    result = await web_tool.ainvoke(tool_args)
                    tool_outputs["web"] = result
                elif tool_name == "RAGSearch":
                    result = await rag_search.ainvoke(tool_args)
                    rag_results["general"] = result
                elif tool_name == "RAGFilteredSearch":
                    result = await rag_filtered_search.ainvoke(tool_args)
                    content_type = tool_args.get("content_type", "unknown")
                    rag_results[content_type] = result
                    
                print(f"{tool_name} returned {len(result) if isinstance(result, list) else 0} results")
                
            except Exception as e:
                print(f"Error executing {tool_name}: {str(e)}")
                continue
    print("Results:", rag_results)
    # Extract visual elements
    try:
        visual_elements = extract_visual_elements(rag_results)
    except Exception as e:
        print(f"Error extracting visual elements: {str(e)}")
        visual_elements = []
    print("\n====== Extracting Visual Elements ======")
    print(f"\nExtracted {len(visual_elements)} visual elements")
    # Create report metadata
    report_metadata = {
        "query": query,
        "timestamp": datetime.now().isoformat(),
        "visual_count": len(visual_elements),
        "sources": {
            "web": len(tool_outputs.get("web", [])),
            "academic": len(tool_outputs.get("arxiv", [])),
            "internal": sum(len(results) for results in rag_results.values())
        }
    }
    
    return {
        **state,
        "tool_outputs": tool_outputs,
        "rag_results": rag_results,
        "visual_elements": visual_elements,
        "report_metadata": report_metadata
    }

async def generate_report(state: AgentState, config: RunnableConfig) -> AgentState:
    """Generate enhanced report with visual elements"""
    print("\n====== Generating Comprehensive Report ======")
    
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    
    # Gather all results
    tool_outputs = state.get("tool_outputs", {})
    rag_results = state.get("rag_results", {})
    visual_elements = state.get("visual_elements", [])
    metadata = state.get("report_metadata", {})
    query = metadata.get("query", "Research Query")
    
    # Format findings
    web_findings = "\n".join([
        f"- {result.get('title')}: {result.get('snippet', '')}"
        for result in tool_outputs.get("web", [])
    ])
    
    arxiv_findings = "\n".join([
        f"- {paper.get('title')}\n  Authors: {', '.join(paper.get('authors', []))}\n  "
        f"Abstract: {paper.get('abstract', '')[:300]}..."
        for paper in tool_outputs.get("arxiv", [])
    ])
    
    rag_findings = ""
    for content_type, results in rag_results.items():
        if content_type not in ['table_image', 'picture', 'chart', 'graph']:
            rag_findings += f"\n\n{content_type.upper()} Results:\n"
            rag_findings += "\n".join([
                f"- Source: {result.get('source')}\n  Content: {result.get('content')[:300]}..."
                for result in results
            ])
    
    # Format visual elements
    visual_section = "\n\n### Visual Elements:\n"
    for visual in visual_elements:
        visual_section += f"""
![{visual['description']}]({visual['url']})
*{visual['description']}* (Source: {visual['source']})
"""
    
    report_prompt = """Create a comprehensive research report in markdown format. 
    
    Requirements:
    1. Use proper markdown formatting including headers, lists, and emphasis
    2. Include and reference all relevant visual elements
    3. Create summary tables where appropriate
    4. Integrate findings from all sources into a coherent narrative
    
    Report Structure:
    
    ID: [Title]    (!!! DONT MISS THIS LINE MUST GENERATE!!!)
    # Research Report: [Title]
    
    ## Executive Summary
    [Concise overview of key findings]
    
    ## Research Context
    [Background and objectives]
    
    ## Methodology
    [Research approach and sources used]
    
    ## Key Findings
    ### External Research
    [Web and academic findings]
    
    ### Internal Knowledge
    [Findings from internal sources]
    
    ### Data Analysis
    [Tables, charts, and visual analysis]
    
    ## Conclusions and Implications
    [Key takeaways and recommendations]
    
    ## Appendix
    ### Sources and References
    [Detailed source list]
    
    
    
    Important:
    - Reference visual elements in the text using proper markdown syntax
    - Use tables to summarize quantitative data
    - Include captions and sources for all visual elements
    - Maintain professional tone and formatting
    - IT MUST BE ONLY MARKDOWN FORMAT. DONT INCLUDE ANY ADDITIONAL TEXT OR FORMATTING LIKE ```MARKDOWN ``` TO SAY IT IS MARKDOWN. IMPORTANT 
    - When referencing any tables or figures, you MUST include them using markdown image syntax: 
            ![description](url). Every visual element you mention must be included with its URL."""
    
    messages = [
        SystemMessage(content=report_prompt),
        HumanMessage(content=f"""
        Query: {query}
        
        Web Research:
        {web_findings}
        
        Academic Research:
        {arxiv_findings}
        
        Internal Knowledge:
        {rag_findings}
        
        Visual Elements:
        {visual_section}
        
        Metadata:
        {metadata}
        """)
    ]
    
    response = await m.ainvoke(messages, config)
    
    return {
        "messages": [response]
    }

def format_report_for_export(report: str, metadata: Dict[str, Any]) -> str:
    """Format the report for export with proper headers and metadata"""
    timestamp = metadata.get("timestamp", datetime.now().isoformat())
    formatted_date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    header = f"""---
Generated: {formatted_date}
Query: {metadata.get('query', 'Research Query')}
Sources: 
  - Web: {metadata.get('sources', {}).get('web', 0)}
  - Academic: {metadata.get('sources', {}).get('academic', 0)}
  - Internal: {metadata.get('sources', {}).get('internal', 0)}
Visual Elements: {metadata.get('visual_count', 0)}
---

"""
    return header + report

# Define the graph
agent = StateGraph(AgentState)

# Add nodes
agent.add_node("research", research_with_tools)
agent.add_node("report", generate_report)

# Set entry point
agent.set_entry_point("research")

# Add edges
agent.add_edge("research", "report")
agent.add_edge("report", END)

# Compile the agent
research_agent = agent.compile(
    checkpointer=MemorySaver(),
)