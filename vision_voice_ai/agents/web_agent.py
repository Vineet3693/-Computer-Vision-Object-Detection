"""
Web Search Agent - Searches the internet using DuckDuckGo
for real-time information and facts
"""
from typing import List, Optional
from duckduckgo_search import DDGS


class WebAgent:
    """
    Web Search Agent responsible for:
    - Searching internet for current information
    - Finding facts not in training data
    - Getting latest news and updates
    - Answering time-sensitive questions
    """

    def __init__(self):
        self.ddgs = DDGS()
        self.max_results = 5

    def search(self, query: str, max_results: Optional[int] = None) -> List[dict]:
        """
        Search the web for information
        
        Args:
            query: Search query
            max_results: Maximum number of results (default: 5)
            
        Returns:
            List of search results with title, href, and body
        """
        if max_results is None:
            max_results = self.max_results
        
        try:
            results = list(self.ddgs.text(query, max_results=max_results))
            return results
        except Exception as e:
            return [{"title": "Search Error", "body": str(e)}]

    def get_answer(self, question: str) -> str:
        """
        Get a concise answer to a question from web search
        
        Args:
            question: User's question
            
        Returns:
            Concise answer suitable for speech
        """
        results = self.search(question, max_results=3)
        
        if not results or 'error' in results[0].get('title', '').lower():
            return "I couldn't find information about that online."
        
        # Extract key information from top results
        answers = []
        for result in results[:2]:
            body = result.get('body', '')
            if body and len(body) > 20:
                answers.append(body)
        
        if not answers:
            return "I found some results but couldn't extract a clear answer."
        
        # Combine top results into coherent response
        combined = " ".join(answers[:2])
        
        # Truncate if too long
        if len(combined) > 300:
            combined = combined[:297] + "..."
        
        return combined

    def get_latest_news(self, topic: str = "AI") -> str:
        """
        Get latest news about a topic
        
        Args:
            topic: News topic
            
        Returns:
            Summary of latest news
        """
        query = f"latest news about {topic}"
        results = self.search(query, max_results=5)
        
        if not results:
            return f"I couldn't find recent news about {topic}."
        
        # Extract headlines
        headlines = []
        for result in results[:3]:
            title = result.get('title', '')
            if title:
                headlines.append(title)
        
        if not headlines:
            return f"No recent news found about {topic}."
        
        response = f"Here are the latest headlines about {topic}: "
        response += ". ".join(headlines) + "."
        
        return response

    def lookup_definition(self, term: str) -> str:
        """
        Look up definition or explanation of a term
        
        Args:
            term: Term to define
            
        Returns:
            Clear definition
        """
        query = f"what is {term}"
        results = self.search(query, max_results=3)
        
        if not results:
            return f"I couldn't find a definition for {term}."
        
        # Try to find a clear definition
        for result in results:
            body = result.get('body', '')
            title = result.get('title', '')
            
            # Prefer results that look like definitions
            if any(word in body.lower() for word in ["refers to", "means", "is a", "defined as"]):
                return body[:200]
        
        # Fallback to first result
        return results[0].get('body', 'Definition not found.')

    def find_how_to(self, task: str) -> str:
        """
        Find instructions for how to do something
        
        Args:
            task: Task or activity
            
        Returns:
            Brief instructions
        """
        query = f"how to {task}"
        results = self.search(query, max_results=5)
        
        if not results:
            return f"I couldn't find instructions for {task}."
        
        # Extract step-by-step info if available
        instructions = []
        for result in results[:2]:
            body = result.get('body', '')
            if body:
                instructions.append(body)
        
        if not instructions:
            return "I found some resources but couldn't extract clear steps."
        
        combined = " ".join(instructions)
        
        # Truncate for speech
        if len(combined) > 250:
            combined = combined[:247] + "..."
        
        return f"Here's what I found: {combined}"

    def verify_fact(self, claim: str) -> str:
        """
        Verify if a claim or fact is accurate
        
        Args:
            claim: Claim to verify
            
        Returns:
            Verification result
        """
        query = f"is it true that {claim}"
        results = self.search(query, max_results=3)
        
        if not results:
            return "I couldn't verify that information."
        
        # Summarize what sources say
        summaries = []
        for result in results:
            title = result.get('title', '')
            if title:
                summaries.append(title)
        
        if summaries:
            return "Sources suggest: " + ". ".join(summaries[:2])
        
        return "I found some information but couldn't clearly verify the claim."

    def get_weather_info(self, location: str = "current location") -> str:
        """
        Get weather information (simulated - would need weather API for real data)
        
        Args:
            location: Location name
            
        Returns:
            Weather information
        """
        # Note: For production, integrate with actual weather API
        # This is a placeholder using web search
        query = f"current weather in {location}"
        results = self.search(query, max_results=2)
        
        if results:
            return results[0].get('body', 'Weather information not available.')
        
        return "I couldn't retrieve current weather information. Please check a weather service."

    def search_with_context(self, query: str, context: str) -> str:
        """
        Search with additional context for better results
        
        Args:
            query: Search query
            context: Additional context
            
        Returns:
            Search result
        """
        enhanced_query = f"{query} {context}"
        return self.get_answer(enhanced_query)
