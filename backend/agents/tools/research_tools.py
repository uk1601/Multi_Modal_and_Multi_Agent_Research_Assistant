import requests
from datetime import datetime
from typing import List, TypedDict, Literal, Dict, Any
from typing import Dict, List, Any
from urllib.parse import quote_plus
from xml.etree import ElementTree
from langchain_core.tools import BaseTool, tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from langgraph.graph import END, MessagesState, StateGraph
from backend.agents.models import models

class ResearchResults(TypedDict):
    web_findings: str
    academic_findings: str
    sources: List[Dict[str, str]]


class ResearchState(MessagesState, total=False):
    """State management for the research agent"""
    query: str
    query_aspects: List[str]
    web_research: ResearchResults | None
    academic_research: ResearchResults | None
    research_stage: Literal["planning", "researching", "synthesizing", "complete"]
    final_response: str | None


@tool("ArXiv")
def fetch_arxiv(query: str) -> List[Dict[str, Any]]:
    """Fetches academic papers from arXiv.
    Args:
        query (str): Search query for papers
    Returns:
        List[Dict]: List of papers with title, authors, abstract, and url
    """
    try:
        # Clean and simplify the query
        keywords = extract_keywords(query)
        sanitized_query = quote_plus(keywords.strip())
        
        print(f"----------------Fetching papers for query-----------: {keywords}")
        
        url = f"http://export.arxiv.org/api/query?search_query=all:{sanitized_query}&start=0&max_results=5"
        
        # Make request
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse XML
        root = ElementTree.fromstring(response.content)
        formatted_papers = []

        # Process each entry
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            try:
                # Extract paper details
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip()
                abstract = entry.find('{http://www.w3.org/2005/Atom}summary').text.strip()
                authors = [
                    author.find('{http://www.w3.org/2005/Atom}name').text.strip()
                    for author in entry.findall('{http://www.w3.org/2005/Atom}author')
                ]
                
                # Get paper URL
                urls = entry.findall('{http://www.w3.org/2005/Atom}link')
                url = next(
                    (link.get('href') for link in urls if link.get('rel') == 'alternate'),
                    None
                )
                
                if url:
                    print(f"URL: {url}")
                
                formatted_papers.append({
                    "title": title,
                    "authors": authors,
                    "abstract": abstract,
                    "url": url
                })
                
            except AttributeError as e:
                print(f"Skipping paper due to missing data: {str(e)}")
                continue

        print(f"length of papers: {len(formatted_papers)}")
        return formatted_papers

    except Exception as e:
        print(f"Error fetching papers: {str(e)}")
        return []


def extract_keywords(query: str) -> str:
    """Extract key search terms from a longer query"""
    # Remove common words and simplify query
    stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
    
    # Split into words and filter
    words = query.lower().split()
    keywords = [word for word in words if word not in stopwords]
    
    # Take only the most relevant terms (up to 5)
    important_terms = keywords[:5]
    
    # Add year constraint for recency
    return " ".join(important_terms) + " year:2023"


@tool("WebSearch")
def web_search(query: str) -> List[Dict[str, str]]:
    """Searches web for relevant information.
    Args:
        query (str): Search query
    Returns:
        List[Dict]: List of search results
    """
    try:
        search = DuckDuckGoSearchResults()
        raw_results = search.run(query)
        
        # Parse the raw string results
        results = []
        current_result = {}
        
        for line in raw_results.split('\n'):
            if line.startswith('title: '):
                if current_result:
                    results.append(current_result.copy())
                current_result = {'title': line[7:]}
            elif line.startswith('link: '):
                current_result['link'] = line[6:]
            elif line.startswith('snippet: '):
                current_result['snippet'] = line[9:]
        
        if current_result:
            results.append(current_result.copy())
            
        return results

    except Exception as e:
        print(f"Web search error: {str(e)}")
        return []


async def academic_research(query: str, config: RunnableConfig) -> ResearchResults:
    """Conduct academic research using arXiv"""
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    
    try:
        # Get papers
        papers = await fetch_arxiv.ainvoke(query)
        
        if not papers:
            return {
                "web_findings": "",
                "academic_findings": f"No academic papers found for: {query}",
                "sources": []
            }
        
        # Format papers for analysis
        formatted_papers = "\n\n".join([
            f"Title: {p.get('title')}\nAuthors: {', '.join(p.get('authors', []))}\n"
            f"Abstract: {p.get('abstract', 'No abstract available')}"
            for p in papers
        ])
        
        # Analyze papers
        analysis_prompt = f"""Analyze these academic papers about {query}.
        Focus on:
        1. Key research findings and contributions
        2. Novel methodologies or approaches
        3. Important conclusions and future directions
        
        Be specific and cite papers by their titles when discussing findings.
        """
        
        messages = [
            SystemMessage(content=analysis_prompt),
            HumanMessage(content=formatted_papers)
        ]
        
        analysis = await m.ainvoke(messages, config)
        
        return {
            "web_findings": "",
            "academic_findings": analysis.content,
            "sources": [
                {
                    "title": p.get("title", ""),
                    "url": p.get("url", ""),
                    "type": "academic",
                    "authors": p.get("authors", []),
                    "abstract": p.get("abstract", "")[:500] + "..."
                }
                for p in papers
            ]
        }
        
    except Exception as e:
        print(f"Academic research error: {str(e)}")
        return {
            "web_findings": "",
            "academic_findings": f"Error in academic research: {str(e)}",
            "sources": []
        }


async def web_research(query: str, config: RunnableConfig) -> ResearchResults:
    """Conduct web research"""
    m = models[config["configurable"].get("model", "gpt-4o-mini")]
    
    try:
        # Get web results
        results = await web_search.ainvoke(query)
        
        if not results:
            return {
                "web_findings": f"No web results found for: {query}",
                "academic_findings": "",
                "sources": []
            }
        
        # Format results for analysis
        formatted_results = "\n\n".join([
            f"Title: {r.get('title')}\nURL: {r.get('link')}\n"
            f"Summary: {r.get('snippet', 'No summary available')}"
            for r in results
        ])
        
        # Analyze results
        analysis_prompt = f"""Analyze these web search results about {query}.
        Focus on:
        1. Key findings and trends
        2. Important developments
        3. Notable insights
        
        Be specific and cite sources by their titles.
        """
        
        messages = [
            SystemMessage(content=analysis_prompt),
            HumanMessage(content=formatted_results)
        ]
        
        analysis = await m.ainvoke(messages, config)
        
        return {
            "web_findings": analysis.content,
            "academic_findings": "",
            "sources": [
                {
                    "title": r.get("title", ""),
                    "url": r.get("link", ""),
                    "type": "web",
                    "snippet": r.get("snippet", "")
                }
                for r in results
            ]
        }
        
    except Exception as e:
        print(f"Web research error: {str(e)}")
        return {
            "web_findings": f"Error in web research: {str(e)}",
            "academic_findings": "",
            "sources": []
        }


# Create tool instances
arxiv_tool = fetch_arxiv

web_tool = web_search
