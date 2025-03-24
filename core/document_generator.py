from typing import List, Optional, Dict
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
        
        # 3. Perform overall document evaluation to identify section-specific issues
        print("\nPerforming overall document evaluation...")
        section_critiques = self._evaluate_document_sections(sections, plan, topic)
        
        # 4. Revise each section based on critiques
        print("\nRevising sections based on evaluation...")
        revised_sections = self._revise_sections(sections, section_critiques, plan)
        
        # 6. Combine all content
        print("\nFinalizing document...")
        document = self._combine_content(revised_sections)
        
        # 7. Save to file if specified
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(document)
            print(f"\nDocument saved to: {output_file}")
        
        return document
    
    def _combine_content(
        self,
        sections: List[GeneratedSection]) -> str:
        """Combine all sections into a complete document."""
        # Combine main content
        content = "# " + sections[0].title + "\n\n"
        content += sections[0].content + "\n\n"
        
        for section in sections[1:-1]:  # Skip intro and conclusion
            content += f"# {section.title}\n\n"
            content += section.content + "\n\n"
        
        content += f"# {sections[-1].title}\n\n"
        content += sections[-1].content + "\n\n"
        
        
        return content
        
    def _evaluate_document_sections(
        self,
        sections: List[GeneratedSection],
        document_plan: DocumentPlan,
        topic: str
    ) -> Dict[int, str]:
        """Evaluate the overall document and create specific critiques for each section."""
        # Create a preview of the entire document (with truncated sections to fit token limits)
        section_previews = []
        for i, section in enumerate(sections):
            # Truncate the content to 200 chars to avoid token limits
            truncated_content = section.content[:300] + "..." if len(section.content) > 300 else section.content
            section_previews.append(f"Section {i}: {section.title}\n{truncated_content}\n")
        
        document_preview = "\n".join(section_previews)
        
        prompt = f"""Evaluate this academic document about "{topic}" and provide specific improvement critiques for EACH SECTION.

Document Plan:
- Introduction: {document_plan.introduction.title}
- Main Sections: {", ".join(s.title for s in document_plan.main_sections)}
- Conclusion: {document_plan.conclusion.title}
- Target Length: {document_plan.total_estimated_length} words

Document Sections Preview:
{document_preview}

For EACH SECTION (0, 1, 2, etc.), provide:
1. Specific issues with coherence, clarity, or style
2. Content gaps or imbalances
3. Academic tone inconsistencies
4. Improvement suggestions for readability
5. Transition improvement recommendations

Format your response as follows:
SECTION 0:
[Critique and improvement suggestions for the introduction]

SECTION 1:
[Critique and improvement suggestions for main section 1]

... and so on for each section ...

Be specific and actionable with your critiques. Each section's feedback will be used to revise that section.
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
                critique_text = self.content_generator._clean_markdown_blocks(response.text)
            else:
                critique_text = response.text
            
            # Parse the critiques by section
            section_critiques = {}
            current_section = None
            current_critique = []
            
            for line in critique_text.split('\n'):
                if line.startswith('SECTION ') and ':' in line:
                    # Save the previous section critique if exists
                    if current_section is not None and current_critique:
                        section_critiques[current_section] = '\n'.join(current_critique)
                        current_critique = []
                    
                    # Extract section number
                    try:
                        section_num = int(line[8:line.find(':')])
                        current_section = section_num
                    except ValueError:
                        continue
                elif current_section is not None:
                    current_critique.append(line)
            
            # Add the last section critique
            if current_section is not None and current_critique:
                section_critiques[current_section] = '\n'.join(current_critique)
            
            return section_critiques
            
        except Exception as e:
            print(f"Error evaluating document sections: {e}")
            return {}
    
    def _revise_sections(
        self,
        sections: List[GeneratedSection],
        section_critiques: Dict[int, str],
        document_plan: DocumentPlan
    ) -> List[GeneratedSection]:
        """Revise each section based on critiques from overall evaluation."""
        revised_sections = []
        
        for i, section in enumerate(sections):
            if i not in section_critiques:
                revised_sections.append(section)  # Keep original if no critique
                continue
                
            print(f"Revising section: {section.title}...")
            critique = section_critiques[i]
            
            # Determine which original section from document plan this is
            if i == 0:
                original_section = document_plan.introduction
            elif i == len(sections) - 1:
                original_section = document_plan.conclusion
            else:
                original_section = document_plan.main_sections[i-1]
            
            # Create a context with previous sections for coherence
            previous_context = []
            if i > 0:
                for prev_i in range(max(0, i-2), i):
                    prev_section = sections[prev_i]
                    # Include a preview of previous sections
                    preview = prev_section.content[:200] + "..." if len(prev_section.content) > 200 else prev_section.content
                    previous_context.append(f"{prev_section.title}:\n{preview}")
            
            # Generate revised content
            revised_section = self._generate_revised_section(
                section, 
                critique, 
                original_section,
                previous_context
            )
            
            revised_sections.append(revised_section)
            
        return revised_sections
    
    def _generate_revised_section(
        self,
        original_section: GeneratedSection,
        critique: str,
        plan_section: Section,
        previous_context: List[str]
    ) -> GeneratedSection:
        """Generate revised content for a section based on critique and original content."""
        context = "\n".join(previous_context) if previous_context else "None"
        
        prompt = f"""Revise the following section of an academic article based on critique feedback:

Section Title: {original_section.title}
Original Content: 
{original_section.content}

Critique and Improvements Needed:
{critique}

Previous Sections Context:
{context}

Section Plan:
- Description: {plan_section.description}
- Subsections: {', '.join(plan_section.subsections)}
- Estimated Length: {plan_section.estimated_length} words

Requirements:
1. Address ALL issues mentioned in the critique
2. Improve readability and academic tone
3. Ensure smooth transitions with previous sections
4. Maintain the original intent and structure
5. Stay within the approximate target length

Format the revised content in Markdown with proper paragraphs.
DO NOT include the title in your response.
IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
"""

        try:
            response = self.content_generator.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.5,
                    "max_output_tokens": 4000,
                }
            )
            
            # Clean any potential markdown code blocks from the response
            revised_content = self.content_generator._clean_markdown_blocks(response.text) if hasattr(self.content_generator, '_clean_markdown_blocks') else response.text
            
            return GeneratedSection(
                title=original_section.title,
                content=revised_content,
                subsections=original_section.subsections
            )
            
        except Exception as e:
            print(f"Error revising section {original_section.title}: {e}")
            return original_section  # Return original if revision fails 