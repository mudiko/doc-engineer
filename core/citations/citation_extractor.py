from typing import Dict, List, Optional
from dataclasses import dataclass
from ..ai import Source


@dataclass
class Citation:
    authors: List[str]
    year: int
    title: str
    venue: str
    doi: str
    page_numbers: Optional[str] = None


class CitationExtractor:
    def __init__(self, citation_style: str = "APA"):
        self.citation_style = citation_style
        self.citation_map: Dict[str, Citation] = {}
        self.citation_counters: Dict[str, int] = {}

    def extract_citation(self, source: Source) -> Citation:
        """Extract citation information from a source."""
        citation = Citation(
            authors=source.authors,
            year=source.year,
            title=source.title,
            venue=source.venue,
            doi=source.doi,
        )

        # Generate unique citation key
        citation_key = self._generate_citation_key(citation)
        self.citation_map[citation_key] = citation

        return citation

    def format_inline_citation(self, citation: Citation) -> str:
        """Format citation for inline use according to the specified style."""
        if self.citation_style == "APA":
            return self._format_apa_inline(citation)
        elif self.citation_style == "MLA":
            return self._format_mla_inline(citation)
        else:
            raise ValueError(f"Unsupported citation style: {self.citation_style}")

    def format_reference(self, citation: Citation) -> str:
        """Format citation for reference list according to the specified style."""
        if self.citation_style == "APA":
            return self._format_apa_reference(citation)
        elif self.citation_style == "MLA":
            return self._format_mla_reference(citation)
        else:
            raise ValueError(f"Unsupported citation style: {self.citation_style}")

    def _generate_citation_key(self, citation: Citation) -> str:
        """Generate a unique key for a citation."""
        authors_str = ", ".join(citation.authors[:2])
        if len(citation.authors) > 2:
            authors_str += " et al."
        return f"{authors_str} ({citation.year})"

    def _format_apa_inline(self, citation: Citation) -> str:
        """Format citation in APA style for inline use."""
        if len(citation.authors) == 1:
            return f"({citation.authors[0]}, {citation.year})"
        elif len(citation.authors) == 2:
            return f"({citation.authors[0]} & {citation.authors[1]}, {citation.year})"
        else:
            return f"({citation.authors[0]} et al., {citation.year})"

    def _format_apa_reference(self, citation: Citation) -> str:
        """Format citation in APA style for reference list."""
        authors = ", ".join(citation.authors)
        if len(citation.authors) > 1:
            authors = authors.replace(", ", ", & ", -1)

        reference = f"{authors} ({citation.year}). {citation.title}. {citation.venue}"
        if citation.doi:
            reference += f". https://doi.org/{citation.doi}"

        return reference

    def _format_mla_inline(self, citation: Citation) -> str:
        """Format citation in MLA style for inline use."""
        if len(citation.authors) == 1:
            return f"({citation.authors[0]} {citation.year})"
        elif len(citation.authors) == 2:
            return f"({citation.authors[0]} and {citation.authors[1]} {citation.year})"
        else:
            return f"({citation.authors[0]} et al. {citation.year})"

    def _format_mla_reference(self, citation: Citation) -> str:
        """Format citation in MLA style for reference list."""
        authors = ", ".join(citation.authors)
        if len(citation.authors) > 1:
            authors = authors.replace(", ", ", and ", -1)

        reference = f'{authors}. "{citation.title}." {citation.venue}, {citation.year}'
        if citation.doi:
            reference += f", https://doi.org/{citation.doi}"

        return reference
