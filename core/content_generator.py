from typing import Dict, Any, List
from dataclasses import dataclass
import google.generativeai as genai
import os
from dotenv import load_dotenv
from .document_planner import Section, DocumentPlan

load_dotenv()


@dataclass
class GeneratedSection:
    title: str
    content: str
    subsections: List[str]


class ContentGenerator:
    def __init__(
        self,
        model_name: str = "gemini-pro",
        search_manager: Any = None,
        citation_extractor: Any = None,
    ):
        self.model_name = model_name
        self.search_manager = search_manager
        self.citation_extractor = citation_extractor
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def generate_section_content(
        self, section: Section, previous_sections: List[GeneratedSection] = None
    ) -> GeneratedSection:
        """Generate content for a specific section."""
        context = self._create_section_context(section, previous_sections)

        prompt = f"""Write the content for the following section of an academic article:

{context}

Requirements:
1. Write approximately {section.estimated_length} words
2. Use academic language and tone
3. Structure the content according to the subsections
4. Maintain consistency with previous sections
5. Use clear topic sentences for each paragraph
6. Include appropriate transitions between ideas

Format the content in Markdown with proper headings and paragraphs.
IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
Do NOT wrap your response in ```markdown or any other code block format."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 4000,
                },
            )

            # Clean any potential markdown code blocks from the response
            cleaned_content = self._clean_markdown_blocks(response.text)

            return GeneratedSection(
                title=section.title, content=cleaned_content, subsections=section.subsections
            )

        except Exception as e:
            print(f"Error generating section content: {e}")
            raise

    def _clean_markdown_blocks(self, text: str) -> str:
        """Remove any markdown code blocks from the text."""
        # Remove code blocks with language specifier
        while "```" in text:
            start = text.find("```")
            # Find the closing ```
            end = text.find("```", start + 3)
            if end > start:
                # Replace the code block with its content (if it's not empty)
                block_content = text[start + 3 : end].strip()
                language_end = block_content.find("\n")
                if language_end > 0:  # There's a language specifier
                    block_content = block_content[language_end:].strip()
                # Replace the code block with its content
                text = text[:start] + block_content + text[end + 3 :]
            else:
                break  # No closing ```, stop processing

        return text

    def check_consistency(
        self, sections: List[GeneratedSection], document_plan: DocumentPlan
    ) -> str:
        """Check consistency across all sections and generate a report."""
        sections_text = "\n\n".join(
            f"Section: {section.title}\n{section.content}" for section in sections
        )

        prompt = f"""Review the following document sections for consistency and quality:

{sections_text}

Check for:
1. Logical flow between sections
2. Consistent terminology and definitions
3. Proper transitions between sections
4. Alignment with the original plan
5. Academic tone and style
6. Completeness of coverage

Provide a detailed report in Markdown format with:
- Overall assessment
- Specific issues found
- Recommendations for improvement
- Missing elements
- Strengths and weaknesses

IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
Do NOT wrap your response in ```markdown or any other code block format."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 2000,
                },
            )

            # Clean any potential markdown code blocks from the response
            return self._clean_markdown_blocks(response.text)

        except Exception as e:
            print(f"Error checking consistency: {e}")
            raise

    def _create_section_context(
        self, section: Section, previous_sections: List[GeneratedSection] = None
    ) -> str:
        """Create context for section generation."""
        context = f"""Section Title: {section.title}
Description: {section.description}
Subsections: {', '.join(section.subsections)}
Target Length: {section.estimated_length} words"""

        if previous_sections:
            context += "\n\nPrevious Sections:\n"
            for prev_section in previous_sections:
                context += f"\n{prev_section.title}:\n{prev_section.content[:500]}..."

        return context
