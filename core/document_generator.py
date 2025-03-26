"""
Document Generator

This module provides the core document generation functionality.
"""

from typing import Optional, List, Tuple

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
        show_tokens: bool = False,
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
            show_tokens (bool): Whether to show token usage information (default: False)

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
        generated_sections: List[GeneratedSection] = []

        for i, section in enumerate(document_plan.sections):  # section is a Section
            print(f"  • {i + 1}/{len(document_plan.sections)}: {section.title}")

            # Generate content for this section
            gen_section = self.content_generator.generate_section_content(
                title, section, generated_sections.copy()
            )

            generated_sections.append(gen_section)

        # Step 4: Generate document-wide critique
        print("[4/5] Evaluating document coherence and quality...")

        # First check for consistency issues between sections
        consistency_issues: List[Tuple[str, str]] = []
        for i, section in enumerate(generated_sections[1:], 1):  # section is a GeneratedSection
            previous_sections = generated_sections[:i]
            # Pass the content and title separately as required by the method
            consistency_report = self.content_generator.check_consistency(
                section.content, section.title, previous_sections
            )
            if consistency_report:
                consistency_issues.append((section.title, consistency_report))

        if consistency_issues:
            print(f"Found {len(consistency_issues)} consistency issues")

        # Generate critiques for the entire document
        print("Generating document-wide critique...")
        section_critiques = self.content_generator.evaluate_document_sections(
            title, generated_sections
        )

        if section_critiques:
            print(f"Generated critiques for {len(section_critiques)} sections")

        # Step 5: Revise and improve each section based on critiques
        print("[5/5] Improving sections based on critique...")

        # Only revise if we have meaningful critiques
        if section_critiques or consistency_issues:
            improved_sections: List[GeneratedSection] = []

            for i, section in enumerate(generated_sections):  # section is a GeneratedSection
                section_critique = section_critiques.get(i, "")

                # Add any relevant consistency issues to the critique
                consistency_critique = ""
                for section_title, issue in consistency_issues:
                    if section_title == section.title:
                        consistency_critique = f"Consistency issues: {issue}"

                # Combine all critique info
                combined_critique = section_critique
                if consistency_critique:
                    combined_critique += "\n\n" + consistency_critique

                # Only revise if we have critique feedback
                if combined_critique.strip():
                    print(f"  • Improving {section.title} based on critique")
                    # Get the section from the document plan
                    plan_section = document_plan.sections[i]

                    # Get context from previous sections
                    previous_context = [s.title + "\n" + s.content[:300] for s in improved_sections]

                    # Revise the section based on critique
                    improved_section = self.content_generator.revise_section(
                        section, combined_critique, plan_section, previous_context
                    )
                    improved_sections.append(improved_section)
                else:
                    # No critique, just keep the original
                    improved_sections.append(section)

            # Replace the sections with improved versions
            generated_sections = improved_sections
            print(
                f"Improved {len(section_critiques) + len(consistency_issues)} sections based on critique"
            )
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
        
        # Display token usage information if available and requested
        if show_tokens and hasattr(self.model_provider, 'input_tokens') and hasattr(self.model_provider, 'output_tokens'):
            # Calculate approximate cost (using current Gemini Pro pricing)
            # Gemini Pro pricing: $0.0000125 / 1K input tokens, $0.0000375 / 1K output tokens
            input_cost = self.model_provider.input_tokens / 1000 * 0.0000125
            output_cost = self.model_provider.output_tokens / 1000 * 0.0000375
            total_cost = input_cost + output_cost
            
            print("\n📊 Token Usage Statistics:")
            print(f"Input tokens: {self.model_provider.input_tokens:,}")
            print(f"Output tokens: {self.model_provider.output_tokens:,}")
            print(f"Total tokens: {self.model_provider.input_tokens + self.model_provider.output_tokens:,}")
            print(f"Total API calls: {self.model_provider.total_api_calls}")
            print(f"Estimated cost: ${total_cost:.4f}")
            print("\nNote: Token counts provide insights into API usage and help optimize prompts.")
            print("Each API call consists of input tokens (your prompts) and output tokens (model's responses).")
            if hasattr(self.model_provider, '_use_new_api') and not self.model_provider._use_new_api:
                print("Note: Token counts are approximate since they're using the legacy API.")
        elif hasattr(self.model_provider, 'input_tokens') and hasattr(self.model_provider, 'output_tokens'):
            # Always show basic token count even if detailed stats are not requested
            total_tokens = self.model_provider.input_tokens + self.model_provider.output_tokens
            print(f"Total tokens used: {total_tokens:,} (use --show-tokens for details)")
            
        print("=== Done ===")

        return formatted_document
