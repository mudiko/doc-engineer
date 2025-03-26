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

        # Define a helper function for ThreadPoolExecutor
        def generate_section_content_task(section_index_and_data):
            index, section = section_index_and_data
            print(f"  • Starting {index + 1}/{len(document_plan.sections)}: {section.title}")

            # Introduction needs no previous context, other sections need only minimal context
            # For sections other than intro, we'll use introduction as context
            context = []
            if index > 0 and index < len(generated_sections):
                # Use introduction as context for all sections
                context = [generated_sections[0]] if generated_sections else []

            # Generate content for this section
            gen_section = self.content_generator.generate_section_content(title, section, context)

            print(f"  • Completed {index + 1}/{len(document_plan.sections)}: {section.title}")
            return index, gen_section

        # Generate introduction first (needed as context for other sections)
        if document_plan.sections:
            print(
                f"  • 1/{len(document_plan.sections)}: {document_plan.sections[0].title} (Introduction)"
            )
            intro_section = self.content_generator.generate_section_content(
                title, document_plan.sections[0], []
            )
            generated_sections = [intro_section]

            # Now generate the rest of the sections in parallel
            remaining_sections = [
                (i, section) for i, section in enumerate(document_plan.sections) if i > 0
            ]

            if remaining_sections:
                # Use ThreadPoolExecutor for parallel processing
                with ThreadPoolExecutor(max_workers=3) as executor:
                    # Submit all tasks
                    future_to_section = {
                        executor.submit(generate_section_content_task, (i, section)): (i, section)
                        for i, section in remaining_sections
                    }

                    # Process results as they complete
                    section_results = []
                    for future in concurrent.futures.as_completed(future_to_section):
                        try:
                            index, section = future.result()
                            section_results.append((index, section))
                        except Exception as e:
                            print(f"Error generating section: {e}")

                    # Sort by original index and add to generated sections
                    section_results.sort(key=lambda x: x[0])
                    for _, section in section_results:
                        generated_sections.append(section)

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
            print(f"Generated critiques for {len(section_critiques)} sections:")
            for i, section in enumerate(generated_sections):
                if i in section_critiques:
                    # Show the first sentence or 100 chars of the critique
                    critique_text = section_critiques[i]
                    first_sentence = (
                        critique_text.split(".")[0] + "..."
                        if "." in critique_text
                        else critique_text[:100] + "..."
                    )
                    print(f"  • Section {i+1}: {section.title}")
                    print(f"    Critique: {first_sentence}")

        # Step 5: Revise and improve each section based on critiques
        print("[5/5] Improving sections based on critique...")

        # Only revise if we have meaningful critiques
        if section_critiques or consistency_issues:
            # Prepare the list of sections to improve
            sections_to_improve = []

            # First identify all sections that need improvement
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

                if combined_critique.strip():
                    sections_to_improve.append((i, section, combined_critique))

            print(f"Improving {len(sections_to_improve)} sections in total")

            # Define a function to improve a section with its critique
            def improve_section_task(item):
                i, section, combined_critique = item
                print(f"  • Starting improvements for: {section.title}")

                # Get the section from the document plan
                plan_section = (
                    document_plan.sections[i] if i < len(document_plan.sections) else None
                )

                # Use introduction as context for improved coherence
                previous_context = []
                if generated_sections and i > 0:
                    intro = generated_sections[0]
                    previous_context = [
                        f"• Introduction: {' '.join(intro.content.split()[:50])}..."
                    ]

                # Revise the section based on critique
                improved_section = self.content_generator.revise_section(
                    section, combined_critique, plan_section, previous_context
                )
                print(f"  • Completed improvements for: {section.title}")
                return i, improved_section

            # Process improvements in parallel
            improved_results = []
            with ThreadPoolExecutor(max_workers=3) as executor:
                # Submit all tasks
                future_to_section = {
                    executor.submit(improve_section_task, item): item
                    for item in sections_to_improve
                }

                # Process results as they complete
                for future in concurrent.futures.as_completed(future_to_section):
                    try:
                        i, improved_section = future.result()
                        improved_results.append((i, improved_section))
                    except Exception as e:
                        idx, section, _ = future_to_section[future]
                        print(f"Error improving section '{section.title}': {e}")

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
