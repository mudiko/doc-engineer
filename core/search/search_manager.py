from typing import List, Dict, Any
from dataclasses import dataclass
import os
import requests
from datetime import datetime, timedelta
from ..ai import Source
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class SearchConfig:
    min_sources: int = 5
    quality_threshold: float = 0.8
    max_age_years: int = 5
    peer_reviewed_only: bool = True
    include_databases: List[str] = None

class SearchManager:
    def __init__(self, config: SearchConfig = None):
        self.config = config or SearchConfig()
        self.available_databases = [
            "PubMed",
            "IEEE",
            "ACM Digital Library",
            "Google Scholar",
            "Web of Science"
        ]
        self.api_keys = {
            "semantic_scholar": os.getenv("SEMANTIC_SCHOLAR_API_KEY"),
            "ieee": os.getenv("IEEE_API_KEY"),
            "acm": os.getenv("ACM_API_KEY")
        }
    
    def search(
        self,
        topic: str,
        min_sources: int = None,
        quality_threshold: float = None
    ) -> List[Source]:
        """
        Search for relevant sources based on the topic and configuration.
        """
        if min_sources:
            self.config.min_sources = min_sources
        if quality_threshold:
            self.config.quality_threshold = quality_threshold
            
        # Search across multiple databases
        all_sources = []
        
        # Search Semantic Scholar
        semantic_sources = self._search_semantic_scholar(topic)
        all_sources.extend(semantic_sources)
        
        # Search IEEE if configured
        if self.config.include_databases and "IEEE" in self.config.include_databases:
            ieee_sources = self._search_ieee(topic)
            all_sources.extend(ieee_sources)
        
        # Search ACM if configured
        if self.config.include_databases and "ACM" in self.config.include_databases:
            acm_sources = self._search_acm(topic)
            all_sources.extend(acm_sources)
        
        # Validate and filter sources
        validated_sources = self._validate_sources(all_sources)
        
        # Ensure minimum number of sources
        if len(validated_sources) < self.config.min_sources:
            print(f"Warning: Only found {len(validated_sources)} valid sources out of {self.config.min_sources} required")
        
        return validated_sources
    
    def _search_semantic_scholar(self, topic: str) -> List[Source]:
        """Search Semantic Scholar API."""
        try:
            headers = {"x-api-key": self.api_keys["semantic_scholar"]} if self.api_keys["semantic_scholar"] else {}
            response = requests.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": topic,
                    "limit": 20,
                    "fields": "title,authors,year,venue,doi,abstract,citationCount"
                },
                headers=headers
            )
            
            if response.status_code == 200:
                papers = response.json().get("data", [])
                return [self._create_source_from_paper(paper) for paper in papers]
            else:
                print(f"Error searching Semantic Scholar: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error in Semantic Scholar search: {e}")
            return []
    
    def _search_ieee(self, topic: str) -> List[Source]:
        """Search IEEE API."""
        # TODO: Implement IEEE API search
        return []
    
    def _search_acm(self, topic: str) -> List[Source]:
        """Search ACM Digital Library API."""
        # TODO: Implement ACM API search
        return []
    
    def _create_source_from_paper(self, paper: Dict[str, Any]) -> Source:
        """Create a Source object from paper data."""
        authors = [author.get("name", "") for author in paper.get("authors", [])]
        return Source(
            authors=authors,
            year=paper.get("year", 0),
            title=paper.get("title", ""),
            venue=paper.get("venue", ""),
            doi=paper.get("doi", ""),
            content=paper.get("abstract", ""),
            credibility_score=self._calculate_credibility_score(paper)
        )
    
    def _calculate_credibility_score(self, paper: Dict[str, Any]) -> float:
        """Calculate credibility score based on various factors."""
        score = 0.0
        
        # Factor 1: Citation count (30%)
        max_citations = 1000  # Normalize to this maximum
        citation_score = min(paper.get("citationCount", 0) / max_citations, 1.0)
        score += 0.3 * citation_score
        
        # Factor 2: Venue quality (30%)
        venue_score = self._assess_venue_quality(paper.get("venue", ""))
        score += 0.3 * venue_score
        
        # Factor 3: Recency (20%)
        year = paper.get("year", 0)
        current_year = datetime.now().year
        age = current_year - year
        recency_score = max(0, 1 - (age / self.config.max_age_years))
        score += 0.2 * recency_score
        
        # Factor 4: Author count (20%)
        author_count = len(paper.get("authors", []))
        author_score = min(author_count / 5, 1.0)  # Normalize to 5 authors
        score += 0.2 * author_score
        
        return score
    
    def _assess_venue_quality(self, venue: str) -> float:
        """Assess the quality of a publication venue."""
        # TODO: Implement venue quality assessment
        # This could use a predefined list of high-quality venues
        # or integrate with journal impact factor data
        return 0.8  # Default score
    
    def _validate_sources(self, sources: List[Source]) -> List[Source]:
        """Validate sources based on configured criteria."""
        validated_sources = []
        
        for source in sources:
            if self._meets_criteria(source):
                validated_sources.append(source)
                
        return validated_sources
    
    def _meets_criteria(self, source: Source) -> bool:
        """Check if a source meets all validation criteria."""
        # Check publication date
        if self.config.max_age_years:
            current_year = datetime.now().year
            if current_year - source.year > self.config.max_age_years:
                return False
            
        # Check credibility score
        if source.credibility_score < self.config.quality_threshold:
            return False
            
        # Check peer-review status if required
        if self.config.peer_reviewed_only:
            # TODO: Implement peer-review verification
            pass
            
        return True
    
    def _extract_metadata(self, raw_source: Dict[str, Any]) -> Source:
        """Extract metadata from raw source data."""
        # TODO: Implement metadata extraction
        pass 