from typing import List, Optional
from .document_planner import DocumentPlanner, DocumentPlan, Section
from .content_generator import ContentGenerator, GeneratedSection
from .search.search_manager import SearchManager, SearchConfig
from .search.citation_extractor import CitationExtractor

class DocumentGenerator:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash-thinking-exp-01-21",
        use_citations: bool = False,
        use_search: bool = False
    ):
        self.model_name = model_name
        self.use_citations = use_citations
        self.use_search = use_search
        
        # Initialize components
        self.planner = DocumentPlanner(model_name)
        self.content_generator = ContentGenerator(
            model_name=model_name,
            search_manager=SearchManager() if use_search else None,
            citation_extractor=CitationExtractor() if use_citations else None
        )
    
    def generate_document(
        self,
        topic: str,
        target_length: int = 5000,
        output_file: Optional[str] = None
    ) -> str:
        """Generate a complete academic document."""
        print(f"Generating document about: {topic}")
        
        # 1. Create document plan
        print("\nCreating document plan...")
        plan = self.planner.create_plan(topic, target_length)
        
        # 2. Generate content for each section
        print("\nGenerating content...")
        sections = []
        
        # Generate introduction
        print(f"Generating {plan.introduction.title}...")
        intro = self.content_generator.generate_section_content(plan.introduction)
        sections.append(intro)
        
        # Generate main sections
        for i, section in enumerate(plan.main_sections, 1):
            print(f"Generating {section.title} ({i}/{len(plan.main_sections)})...")
            content = self.content_generator.generate_section_content(
                section,
                previous_sections=sections
            )
            sections.append(content)
        
        # Generate conclusion
        print(f"Generating {plan.conclusion.title}...")
        conclusion = self.content_generator.generate_section_content(
            plan.conclusion,
            previous_sections=sections
        )
        sections.append(conclusion)
        
        # 3. Check consistency
        print("\nChecking document consistency...")
        consistency_report = self.content_generator.check_consistency(sections, plan)
        
        # 4. Combine all content
        print("\nFinalizing document...")
        document = self._combine_content(sections, consistency_report)
        
        # 5. Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(document)
            print(f"\nDocument saved to: {output_file}")
        
        return document
    
    def _combine_content(
        self,
        sections: List[GeneratedSection],
        consistency_report: str
    ) -> str:
        """Combine all sections into a complete document."""
        # Combine main content
        content = "# " + sections[0].title + "\n\n"
        content += sections[0].content + "\n\n"
        
        for section in sections[1:-1]:  # Skip intro and conclusion
            content += f"# {section.title}\n\n"
            content += section.content + "\n\n"
        
        content += f"# {sections[-1].title}\n\n"
        content += sections[-1].content + "\n\n"
        
        # Add consistency report
        content += "# Document Review\n\n"
        content += consistency_report
        
        return content 