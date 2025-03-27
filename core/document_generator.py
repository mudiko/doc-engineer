"""
Document Generator

This module provides the core document generation functionality.
"""

from typing import Optional, List, Tuple, Dict
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor

from core.modules.content_generator import ContentGenerator, GeminiProvider, ModelProvider
from core.modules.document_parser import Section, DocumentPlan, GeneratedSection
from core.modules.templates import get_template


class DocumentGenerator:
    """Class for generating complete documents with AI assistance."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-2.0-flash-thinking-exp-01-21",
        model_provider: Optional[ModelProvider] = None,
    ):
        """
        Initialize the DocumentGenerator.

        Args:
            api_key (Optional[str]): API key for the AI model service (not needed if model_provider is provided)
            model_name (str): Name of the model to use
            model_provider (Optional[ModelProvider]): A pre-configured model provider (overrides api_key and model_name)
        """
        # Initialize core components
        if model_provider:
            self.model_provider = model_provider
        else:
            self.model_provider = GeminiProvider(api_key=api_key, model_name=model_name)

        self.content_generator = ContentGenerator(self.model_provider)
        print("Document generator initialized successfully")

    def generate_document(
        self,
        title: str,
        num_sections: int = 5,
        template_name: str = "academic",
        output_format: str = "markdown",
        output_path: Optional[str] = None,
        target_length_words: Optional[int] = None,
        show_tokens: bool = True,
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

        Returns:
            str: The generated document content
        """
        print("=== Generating Document ===")

        # Step 1: Create a document plan
        print("[1/5] Creating document plan...")
        document_plan = self.content_generator.create_document_plan(
            title, num_sections, target_length_words=target_length_words
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

        # Step 2: Apply template
        print(f"[2/5] Using '{template_name}' template")
        template = get_template(template_name)

        # Step 3: Generate content for each section
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

            # Generate content for this section
            gen_section = self.content_generator.generate_section_content(title, section, context)

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

        # Step 4: Evaluate document coherence and quality
        print("[4/5] Evaluating document coherence and quality...")
        critiques = self.content_generator.evaluate_document_sections(title, generated_sections)

        issues_found = sum(1 for c in critiques.values() if c.strip())
        if issues_found:
            print(f"Found {issues_found} consistency issues")
            print("Generating document-wide critique...")
            document_critique = self.content_generator.generate_document_critique(
                title, generated_sections
            )

            # Step 5: Improve sections based on critique
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

        # Display token usage information if available
        if (
            show_tokens
            and hasattr(self.model_provider, "input_tokens")
            and hasattr(self.model_provider, "output_tokens")
        ):
            print("\n📊 Token Usage Statistics:")
            print(f"Input tokens: {self.model_provider.input_tokens:,}")
            print(f"Output tokens: {self.model_provider.output_tokens:,}")
            print(
                f"Total tokens: {self.model_provider.input_tokens + self.model_provider.output_tokens:,}"
            )
            print(f"Total API calls: {self.model_provider.total_api_calls}")
            print("\nNote: Token counts provide insights into API usage and help optimize prompts.")
            print(
                "Each API call consists of input tokens (your prompts) and output tokens (model's responses)."
            )
            if (
                hasattr(self.model_provider, "_use_new_api")
                and not self.model_provider._use_new_api
            ):
                print("Note: Token counts are approximate since they're using the legacy API.")
        elif (
            not show_tokens
            and hasattr(self.model_provider, "input_tokens")
            and hasattr(self.model_provider, "output_tokens")
        ):
            # Show minimal token info when show_tokens is False
            total_tokens = self.model_provider.input_tokens + self.model_provider.output_tokens
            print(f"Total tokens used: {total_tokens:,} (use without --hide-tokens to see details)")

        print("=== Done ===")

        return formatted_document
