"""
Document Generator

Main module for generating complete documents with AI assistance.
"""

import os
import sys
import uuid
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
import time
import datetime

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Updated imports after refactoring
from core.generation.content_generator import ContentGenerator
from core.planning.document_parser import Section, DocumentPlan, GeneratedSection
from core.planning.document_planner import DocumentPlanner # Added import
from core.formatting.templates import get_template
from core.citations.citation_manager import CitationManager


class DocumentGenerator:
    """Class for generating complete documents with AI assistance."""

    def __init__(self,
                 document_planner: DocumentPlanner,
                 content_generator: ContentGenerator,
                 citation_manager: CitationManager):
        """
        Initialize the document generator with injected dependencies.

        Args:
            document_planner: An instance of DocumentPlanner.
            content_generator: An instance of ContentGenerator.
            citation_manager: An instance of CitationManager.
        """
        self.document_planner = document_planner
        self.content_generator = content_generator
        self.citation_manager = citation_manager
        # Note: The 'mock' status is now determined by the type of providers injected,
        # rather than a flag within DocumentGenerator.

        print("Document generator initialized successfully with injected dependencies.")

    def generate_document(
        self,
        title: str,
        num_sections: int = 5,
        template_name: str = "academic",
        output_format: str = "markdown",
        output_path: Optional[str] = None,
        target_length_words: Optional[int] = None,
        show_tokens: bool = True,
        with_citations: bool = False,
    ) -> str:
        """
        Generate a complete document based on the given title.

        Args:
            title (str): The title of the document
            num_sections (int): The number of sections to generate (default: 5)
            template_name (str): The template to use (default: "academic")
            output_format (str): Format to output the document in (default: "markdown")
            output_path (Optional[str]): Path to save the document (default: None)
            target_length_words (Optional[int]): Target document length in words (default: None)
            show_tokens (bool): Whether to show token usage information (default: True)
            with_citations (bool): Whether to include citations from academic papers (default: False)

        Returns:
            str: The generated document content
        """
        print("=== Generating Document ===")
        
        # Generate a unique ID for this document
        document_id = str(uuid.uuid4())[:8]
        
        # Step 1: Search for relevant citations if requested
        citations = []
        if with_citations:
            print("[+] Searching for relevant citations...")
            try:
                citations = self.citation_manager.search_papers(
                    topic=title,
                    document_id=document_id,
                    limit=20,
                    limit_per_database=5,
                    # use_mock=self.mock # Removed: Mock status handled by injected CitationManager config
                )
                print(f"Found {len(citations)} relevant citations")
            except Exception as e:
                print(f"Warning: Failed to retrieve citations: {e}")
                print("Proceeding without citations")
        
        # Step 2: Create a document plan using the injected planner
        print("[1/5] Creating document plan...")
        document_plan = self.document_planner.create_plan(
            topic=title, num_sections=num_sections, target_length_words=target_length_words
        )
        print(f"Plan created with {len(document_plan.sections)} sections")

        # Display estimated length information if target length was specified
        if target_length_words:
            print(
                f"Target length: {target_length_words} words (≈{target_length_words / 500:.1f} pages)"
            )
            print(f"Planned length: {document_plan.total_estimated_length} words")

            # Show section breakdown
            print("Section breakdown:")
            print(f"  - Introduction: {document_plan.introduction.estimated_length} words")
            for i, section in enumerate(document_plan.main_sections):
                print(f"  - {section.title}: {section.estimated_length} words")
            print(f"  - Conclusion: {document_plan.conclusion.estimated_length} words")

        # Step 3: Apply template
        print(f"[2/5] Using '{template_name}' template")
        template = get_template(template_name)

        # Step 4: Generate content for each section
        print("[3/5] Generating content...")
        
        # First, generate an abstract section
        abstract_section = Section(
            title="Abstract",
            description="A concise summary of the document",
            subsections=["Summary"],
            estimated_length=150,
            level=2
        )
        
        # Use introduction as context for generating the abstract (if it will be generated first)
        abstract_gen_section = None
        
        # Define a helper function for ThreadPoolExecutor
        def generate_section_content_task(section_index_and_data):
            index, section = section_index_and_data
            section_number = index + 1
            total_sections = len(document_plan.sections) + 1  # +1 for abstract
            
            if index == 0 and section.title == "Abstract":
                print(f"  • Starting {section_number}/{total_sections}: {section.title}")
                gen_section = self.content_generator.generate_section_content(title, section, [])
                print(f"  • Completed {section_number}/{total_sections}: {section.title}")
                return index, gen_section
                
            # For regular sections
            print(f"  • Starting {section_number}/{total_sections}: {section.title}")

            # Introduction needs no previous context, other sections need only minimal context
            context = []
            if index > 0 and len(generated_sections) > 0:
                # Include abstract and introduction as context for other sections
                context = [s for s in generated_sections if s.title in ["Abstract", "Introduction"]]
            
            # Include citation information if available
            citation_context = ""
            if with_citations and citations:
                # Select relevant citations for this section using the CitationManager
                section_citations = self.citation_manager.select_citations_for_section(citations, section.title, section.description, document_id)
                if section_citations:
                    citation_context = "\nRelevant citations for this section:\n"
                    for i, citation in enumerate(section_citations[:5]):  # Limit to 5 citations
                        authors = ", ".join(citation.get("authors", [])[:3])
                        if len(citation.get("authors", [])) > 3:
                            authors += " et al."
                        citation_context += f"[{i+1}] {citation.get('title', 'Untitled')} ({citation.get('year', '')}). {authors}.\n"
                        
                        # Add text chunks if available from vector search
                        if 'text_chunks' in citation and citation['text_chunks']:
                            citation_context += "Relevant excerpts:\n"
                            for j, chunk in enumerate(citation['text_chunks'][:2]):  # Limit to 2 chunks per citation
                                # Trim the chunk if it's too long
                                max_chunk_length = 500
                                if len(chunk) > max_chunk_length:
                                    chunk = chunk[:max_chunk_length] + "..."
                                citation_context += f"  - {chunk}\n\n"
            
            # Generate content for this section
            gen_section = self.content_generator.generate_section_content(
                title, section, context, citation_context=citation_context if citation_context else None
            )

            print(f"  • Completed {section_number}/{total_sections}: {section.title}")
            return index, gen_section

        # Generate the abstract first
        abstract_result = generate_section_content_task((0, abstract_section))
        abstract_gen_section = abstract_result[1]
        
        # Generate all other sections sequentially
        generated_sections = [abstract_gen_section]  # Start with abstract
        
        # Process the regular sections from the plan
        for i, section in enumerate(document_plan.sections):
            result = generate_section_content_task((i + 1, section))
            generated_sections.append(result[1])

        # Step 5: Evaluate document coherence and quality
        print("[4/5] Evaluating document coherence and quality...")
        critiques = self.content_generator.evaluate_document_sections(title, generated_sections)

        issues_found = sum(1 for c in critiques.values() if c.strip())
        if issues_found:
            print(f"Found {issues_found} consistency issues")
            print("Generating document-wide critique...")
            document_critique = self.content_generator.generate_document_critique(
                title, generated_sections
            )

            # Step 6: Improve sections based on critique
            print("[5/5] Improving sections based on critique...")
            sections_to_improve = []
            for i, (section_title, critique) in enumerate(critiques.items()):
                if critique.strip():
                    # Find the section by title
                    section_index = next(
                        (i for i, s in enumerate(generated_sections) if s.title == section_title),
                        None,
                    )
                    if section_index is not None:
                        sections_to_improve.append((section_index, critique))

            print(f"Improving {len(sections_to_improve)} sections in total")
            improved_results = []

            # Improve sections one by one
            for section_index, critique in sections_to_improve:
                section = generated_sections[section_index]
                section_plan = None
                
                # Skip improving abstract if it's included in sections to improve
                if section.title == "Abstract":
                    continue
                    
                # Find the corresponding plan section
                if section.title == "Introduction":
                    section_plan = document_plan.introduction
                elif section.title == "Conclusion":
                    section_plan = document_plan.conclusion
                else:
                    section_plan = next(
                        (s for s in document_plan.main_sections if s.title == section.title), None
                    )

                if not section_plan:
                    print(f"Warning: Could not find plan for section '{section.title}'")
                    continue

                # Use abstract and introduction as context for improvements
                context = [
                    s.content
                    for s in generated_sections
                    if s.title in ["Abstract", "Introduction"] and s.title != section.title
                ]

                print(f"  • Starting improvements for: {section.title}")
                improved_section = self.content_generator.revise_section(
                    section, critique, section_plan, context
                )
                print(f"  • Completed improvements for: {section.title}")

                improved_results.append((section_index, improved_section))

            # Now create the final list of sections with improvements applied
            improved_sections = list(generated_sections)  # Start with a copy of all sections
            for i, improved_section in improved_results:
                improved_sections[i] = improved_section  # Replace improved sections

            # Replace the sections with improved versions
            generated_sections = improved_sections
            print(f"Completed improvements for {len(sections_to_improve)} sections")
        else:
            print("No issues found - document is already well-structured and coherent")

        # Format the document
        print(f"Formatting with {template_name} template...")
        
        # Update the template with citation information if available
        if with_citations and citations:
            bibtex_path = self.citation_manager.get_bibtex_path(document_id)
            citation_keys = self.citation_manager.get_citation_keys(document_id)
            citation_summaries = self.citation_manager.get_citations_summary(document_id)
            
            # Get formatted bibliography if available
            formatted_bibliography = None
            if hasattr(self.citation_manager, 'format_bibliography'):
                formatted_bibliography = self.citation_manager.format_bibliography(document_id)
            
            # Format the document with citations
            formatted_document = template.format_document(
                title=title, 
                sections=generated_sections, 
                output_format=output_format,
                citation_keys=citation_keys,
                citation_summaries=citation_summaries,
                bibtex_path=bibtex_path,
                formatted_bibliography=formatted_bibliography
            )
        else:
            # Format without citations
            formatted_document = template.format_document(
                title=title, sections=generated_sections, output_format=output_format
            )

        # Save document if output path is provided
        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(formatted_document)
            print(f"Document saved to {output_path}")

        word_count = sum(len(s.content.split()) for s in generated_sections)
        print(f"Generated {len(generated_sections)} sections with ~{word_count} words")

        # Show token usage statistics
        if show_tokens:
            input_tokens = self.content_generator.total_input_tokens
            output_tokens = self.content_generator.total_output_tokens
            total_tokens = input_tokens + output_tokens
            total_calls = self.content_generator.total_api_calls

            print("\n📊 Token Usage Statistics:")
            print(f"Input tokens: {input_tokens:,}")
            print(f"Output tokens: {output_tokens:,}")
            print(f"Total tokens: {total_tokens:,}")
            print(f"Total API calls: {total_calls}")
            print(
                "\nNote: Token counts provide insights into API usage and help optimize prompts."
            )
            print(
                "Each API call consists of input tokens (your prompts) and output tokens (model's responses)."
            )

        print("=== Done ===")

        return formatted_document
    
    # Removed _select_citations_for_section method as it's now in CitationManager
