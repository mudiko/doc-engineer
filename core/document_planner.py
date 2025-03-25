from typing import List, Dict, Any
from dataclasses import dataclass
import google.generativeai as genai
import os
import json
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Section:
    title: str
    description: str
    subsections: List[str]
    estimated_length: int  # in words

    def to_dict(self) -> Dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "subsections": self.subsections,
            "estimated_length": self.estimated_length,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Section":
        return cls(
            title=data["title"],
            description=data["description"],
            subsections=data["subsections"],
            estimated_length=data["estimated_length"],
        )


@dataclass
class DocumentPlan:
    topic: str
    introduction: Section
    main_sections: List[Section]
    conclusion: Section
    total_estimated_length: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "introduction": self.introduction.to_dict(),
            "main_sections": [section.to_dict() for section in self.main_sections],
            "conclusion": self.conclusion.to_dict(),
            "total_estimated_length": self.total_estimated_length,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentPlan":
        return cls(
            topic=data["topic"],
            introduction=Section.from_dict(data["introduction"]),
            main_sections=[Section.from_dict(section) for section in data["main_sections"]],
            conclusion=Section.from_dict(data["conclusion"]),
            total_estimated_length=data["total_estimated_length"],
        )


class DocumentPlanner:
    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)

    def create_plan(self, topic: str, target_length: int = 5000) -> DocumentPlan:
        """Create a detailed document plan with sections and subsections."""
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
        "estimated_length": 500
    }},
    "main_sections": [
        {{
            "title": "Section 1",
            "description": "Brief description",
            "subsections": ["subsection 1", "subsection 2", "subsection 3"],
            "estimated_length": 1000
        }},
        {{
            "title": "Section 2",
            "description": "Brief description",
            "subsections": ["subsection 1", "subsection 2", "subsection 3"],
            "estimated_length": 1000
        }}
    ],
    "conclusion": {{
        "title": "Conclusion",
        "description": "Brief description",
        "subsections": ["subsection 1", "subsection 2", "subsection 3"],
        "estimated_length": 500
    }},
    "total_estimated_length": {target_length}
}}

Document requirements:
1. Include 4-5 main sections
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
            return DocumentPlan.from_dict(
                {
                    "topic": topic,
                    "introduction": plan_data["introduction"],
                    "main_sections": plan_data["main_sections"],
                    "conclusion": plan_data["conclusion"],
                    "total_estimated_length": plan_data["total_estimated_length"],
                }
            )

        except Exception as e:
            print(f"Error creating document plan: {e}")
            print(f"Raw response: {response.text if 'response' in locals() else 'No response'}")
            raise

    def _extract_json(self, text: str) -> str:
        """Extract a JSON string from text, handling common formatting issues."""
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

        # Basic cleanup
        text = text.replace("\n", " ").replace("\r", "")

        # Simple smart quotes replacement
        text = text.replace('"', '"').replace('"', '"')
        text = text.replace(""", "'").replace(""", "'")

        # If the text doesn't start with {, attempt to find JSON
        if not text.startswith("{"):
            start = text.find("{")
            if start >= 0:
                end = text.rfind("}")
                if end > start:
                    text = text[start : end + 1]
                else:
                    # No closing brace found, add it
                    text = text[start:] + "}"

        # Fix incomplete JSON by counting opening and closing braces
        open_braces = text.count("{")
        close_braces = text.count("}")
        if open_braces > close_braces:
            # Add missing closing braces
            text += "}" * (open_braces - close_braces)

        # Handle missing comma in the conclusion section (common issue)
        if '"conclusion"' in text and '"total_estimated_length"' not in text:
            # Find the last property in conclusion
            if '"estimated_length": ' in text:
                length_pos = text.rfind('"estimated_length": ')
                number_end = length_pos
                while (
                    number_end < len(text)
                    and text[number_end].isdigit()
                    or text[number_end] in " \t"
                ):
                    number_end += 1
                if number_end < len(text):
                    # Add closing braces for conclusion and the entire object
                    text = text[:number_end] + "}" + text[number_end:]

        return text
