from typing import List, Dict, Any, Optional
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

# Updated import after refactoring
from core.planning.document_parser import Section, DocumentPlan

load_dotenv()


class DocumentPlanner:
    def __init__(self, model_name: str = "gemini-2.0-flash"):
        self.model_name = model_name
        # Ensure API key is configured before creating the model
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            # Handle missing API key gracefully, maybe raise an error or use a mock
            print("Warning: GOOGLE_API_KEY not found. Planner might not function correctly.")
            self.model = None  # Or initialize a mock model
        else:
            genai.configure(api_key=api_key)
            try:
                self.model = genai.GenerativeModel(model_name)
            except Exception as e:
                print(f"Error initializing Gemini model for planner: {e}")
                self.model = None  # Fallback if model init fails

    def create_plan(
        self, topic: str, num_sections: int = 5, target_length_words: Optional[int] = None
    ) -> DocumentPlan:
        """
        Create a detailed document plan with sections and subsections.
        Uses the configured AI model and falls back to a default plan on error.

        Args:
            topic: The topic of the document.
            num_sections: The desired number of main sections (used in fallback).
            target_length_words: The target total word count for the document.

        Returns:
            A DocumentPlan object.
        """
        target_length = target_length_words or 4000  # Default target length if not provided

        # Check if the model was initialized successfully
        if not self.model:
            print("AI model not available for planning. Falling back to default plan.")
            return self._create_default_document_plan(topic, num_sections, target_length)

        prompt = f"""Create a detailed academic document plan for a {target_length}-word article about {topic}.

CRITICAL: Your response MUST be ONLY a valid JSON object with NO additional text or markdown formatting.

STRICT JSON FORMAT REQUIREMENTS:
1. Use standard double quotes (") for all strings and keys
2. Do not use single quotes (')
3. Do not use smart/curly quotes (", ", ', ')
4. Include commas between properties but NOT after the last property in an object or array
5. Ensure all JSON syntax is valid and parseable

The JSON object must have exactly this structure:
{{
    "introduction": {{
        "title": "Introduction",
        "description": "Brief description of what will be covered",
        "subsections": ["subsection 1", "subsection 2", "subsection 3"],
        "estimated_length": {int(target_length * 0.1)}
    }},
    "main_sections": [
        {{
            "title": "Section 1 Title",
            "description": "Brief description",
            "subsections": ["subsection 1", "subsection 2", "subsection 3"],
            "estimated_length": {int((target_length * 0.8) / num_sections)}
        }},
        // ... include {num_sections -1} more main sections ...
    ],
    "conclusion": {{
        "title": "Conclusion",
        "description": "Brief description",
        "subsections": ["subsection 1", "subsection 2", "subsection 3"],
        "estimated_length": {int(target_length * 0.1)}
    }},
    "total_estimated_length": {target_length}
}}

Document requirements:
1. Include {num_sections} main sections
2. Each section should have 3-4 subsections
3. The total_estimated_length should sum to exactly {target_length}

YOUR RESPONSE MUST BE ONLY THE JSON OBJECT WITH NO ADDITIONAL TEXT."""

        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2000,
                },
            )

            # Attempt to extract a valid JSON string
            content = self._extract_json(response.text)

            # Parse the JSON into a dictionary
            plan_data = json.loads(content)

            # Validate the structure
            required_keys = [
                "introduction",
                "main_sections",
                "conclusion",
                "total_estimated_length",
            ]
            for key in required_keys:
                if key not in plan_data:
                    raise ValueError(f"Missing required key: {key}")

            # Create DocumentPlan object using from_dict
            # Ensure topic is included when calling from_dict
            return DocumentPlan.from_dict(plan_data, topic=topic)

        except Exception as e:
            print(f"Error creating document plan with API: {e}")
            print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
            # Fallback to default plan creation
            print("Falling back to default document plan creation.")
            return self._create_default_document_plan(topic, num_sections, target_length)

    def _create_default_document_plan(
        self, title: str, num_sections: int = 5, target_length: int = 4000
    ) -> DocumentPlan:
        """Create a default document plan when the normal creation fails."""
        # Calculate section lengths
        intro_length = int(target_length * 0.1)
        conclusion_length = int(target_length * 0.1)
        # Ensure main_section_length is at least 1
        main_section_length = max(
            1, int((target_length - intro_length - conclusion_length) / num_sections)
        )
        # Adjust total length if rounding caused issues
        calculated_total = intro_length + conclusion_length + (main_section_length * num_sections)
        if calculated_total != target_length:
            # Adjust the last main section length to match the target
            diff = target_length - calculated_total
            # Ensure the adjustment doesn't make the last section length negative
            if main_section_length + diff > 0:
                main_section_lengths = [main_section_length] * (num_sections - 1) + [
                    main_section_length + diff
                ]
            else:  # Distribute difference more evenly if adjustment is too large
                main_section_lengths = [main_section_length] * num_sections
                # Add difference to first section if possible
                main_section_lengths[0] = max(1, main_section_lengths[0] + diff)
            target_length = (
                intro_length + conclusion_length + sum(main_section_lengths)
            )  # Recalculate total
        else:
            main_section_lengths = [main_section_length] * num_sections

        # Create introduction
        introduction = Section(
            title="Introduction",
            description="An introduction to the topic.",
            subsections=["Background", "Overview", "Scope"],
            estimated_length=intro_length,
            level=1,
        )

        # Create main sections
        main_sections = []
        for i in range(num_sections):
            main_sections.append(
                Section(
                    title=f"Section {i + 1}",
                    description=f"Content for section {i + 1}.",
                    subsections=[f"Subsection {i + 1}.{j + 1}" for j in range(3)],
                    estimated_length=main_section_lengths[i],
                    level=1,
                )
            )

        # Create conclusion
        conclusion = Section(
            title="Conclusion",
            description="A conclusion to the topic.",
            subsections=["Summary", "Implications", "Future Directions"],
            estimated_length=conclusion_length,
            level=1,
        )

        # Create and return the document plan
        return DocumentPlan(
            topic=title,
            introduction=introduction,
            main_sections=main_sections,
            conclusion=conclusion,
            total_estimated_length=target_length,
        )

    def _extract_json(self, text: str) -> str:
        """Extract a JSON string from text, handling common formatting issues."""
        import re  # Moved import here as it's only used in this method

        # Clean up the text
        text = text.strip()

        # Handle nested markdown blocks (```markdown with ```json inside)
        if "```markdown" in text and "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        # Handle code blocks with json language specifier
        elif "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        # Handle just code blocks without language specifier
        elif text.startswith("```") and text.endswith("```"):
            text = text[3:-3].strip()

        # Basic cleanup - remove newlines within the JSON structure attempt
        # text = text.replace("\n", " ").replace("\r", "") # Be careful with this, might break valid newlines in strings

        # Simple smart quotes replacement
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace("'", "'").replace(
            "'", "'"
        )  # Keep single quotes for now, json.loads handles them

        # If the text doesn't start with {, attempt to find the first { and last }
        if not text.startswith("{"):
            start = text.find("{")
            if start != -1:
                end = text.rfind("}")
                if end > start:
                    text = text[start : end + 1]
                else:
                    # If no closing brace, maybe it's just truncated? Return what we found.
                    text = text[start:]
            else:
                # No opening brace found at all
                raise ValueError("No JSON object found in the response text.")

        # Attempt to fix incomplete JSON by ensuring brace balance (simple approach)
        open_braces = text.count("{")
        close_braces = text.count("}")
        if open_braces > close_braces:
            text += "}" * (open_braces - close_braces)
        elif close_braces > open_braces:
            # Too many closing braces? Try removing from the end. Risky.
            pass  # Or maybe find the first '{' and last '}' more reliably

        # It's generally better to let json.loads handle final validation
        # than to apply complex regex fixes that might break valid JSON.
        return text
