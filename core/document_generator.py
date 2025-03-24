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
        
        # 4. Perform overall document evaluation
        print("\nPerforming overall document evaluation...")
        overall_evaluation = self._evaluate_overall_document(sections, plan, topic)
        
        # 5. Combine all content
        print("\nFinalizing document...")
        document = self._combine_content(sections, consistency_report, overall_evaluation)
        
        # 6. Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(document)
            print(f"\nDocument saved to: {output_file}")
        
        return document
    
    def _combine_content(
        self,
        sections: List[GeneratedSection],
        consistency_report: str,
        overall_evaluation: str
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
        content += consistency_report + "\n\n"
        
        # Add overall evaluation
        content += "# Overall Document Evaluation\n\n"
        content += overall_evaluation
        
        return content
        
    def _evaluate_overall_document(
        self,
        sections: List[GeneratedSection],
        document_plan: DocumentPlan,
        topic: str
    ) -> str:
        """Evaluate the overall document for consistency, structure, and completeness."""
        # Create a condensed version of each section to avoid token limits
        section_summaries = []
        for section in sections:
            # Create a summary of each section by taking the first 150 chars and last 150 chars
            content = section.content
            summary = f"Section: {section.title}\n"
            summary += f"Beginning: {content[:150]}...\n"
            summary += f"End: ...{content[-150:]}\n"
            section_summaries.append(summary)
        
        section_text = "\n\n".join(section_summaries)
        
        prompt = f"""Evaluate this academic document about "{topic}" for overall consistency and structure.

Document Plan:
- Introduction: {document_plan.introduction.title}
- Main Sections: {", ".join(s.title for s in document_plan.main_sections)}
- Conclusion: {document_plan.conclusion.title}
- Target Length: {document_plan.total_estimated_length} words

Section samples:
{section_text}

Evaluate the document on:
1. Overall narrative flow and coherence
2. Transitions between sections
3. Adherence to academic tone and style
4. Balanced coverage of the topic
5. Potential gaps or redundancies
6. Structural organization
7. Key areas for improvement if specific sections were to be revised

Provide specific comments that could guide future revisions of individual sections.
IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
"""

        try:
            response = self.content_generator.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 2000,
                }
            )
            
            # Clean any potential markdown code blocks from the response
            if hasattr(self.content_generator, '_clean_markdown_blocks'):
                return self.content_generator._clean_markdown_blocks(response.text)
            return response.text
            
        except Exception as e:
            print(f"Error generating overall evaluation: {e}")
            return "Error occurred while generating the overall document evaluation." 