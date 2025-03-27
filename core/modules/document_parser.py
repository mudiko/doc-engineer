from typing import Dict, List, Any, Optional
import json
import re
from dataclasses import dataclass


@dataclass
class Section:
    """Represents a section in a document structure."""

    title: str
    description: str
    subsections: List[str]
    estimated_length: int  # in words
    level: int = 1  # Default level is 1 (main section)

    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary representation."""
        return {
            "title": self.title,
            "description": self.description,
            "subsections": self.subsections,
            "estimated_length": self.estimated_length,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Section":
        """Create a Section from dictionary data."""
        return cls(
            title=data["title"],
            description=data["description"],
            subsections=data["subsections"],
            estimated_length=data["estimated_length"],
            level=data.get("level", 1),  # Default to level 1 if not specified
        )

    @classmethod
    def create_abstract(cls, estimated_length: int = 150) -> "Section":
        """Create an abstract section."""
        return cls(
            title="Abstract",
            description="A concise summary of the document",
            subsections=["Summary"],
            estimated_length=estimated_length,
            level=2
        )


@dataclass
class DocumentPlan:
    """Represents the overall plan and structure of a document."""

    topic: str
    introduction: Section
    main_sections: List[Section]
    conclusion: Section
    total_estimated_length: int

    @property
    def sections(self) -> List[Section]:
        """Get a combined list of all sections (introduction, main sections, conclusion)."""
        return [self.introduction] + self.main_sections + [self.conclusion]

    def to_dict(self) -> Dict[str, Any]:
        """Convert document plan to dictionary representation."""
        return {
            "topic": self.topic,
            "introduction": self.introduction.to_dict(),
            "main_sections": [section.to_dict() for section in self.main_sections],
            "conclusion": self.conclusion.to_dict(),
            "total_estimated_length": self.total_estimated_length,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any], topic: str) -> "DocumentPlan":
        """Create a DocumentPlan from dictionary data."""
        return cls(
            topic=topic,
            introduction=Section.from_dict(data["introduction"]),
            main_sections=[Section.from_dict(section) for section in data["main_sections"]],
            conclusion=Section.from_dict(data["conclusion"]),
            total_estimated_length=data["total_estimated_length"],
        )


@dataclass
class GeneratedSection:
    """Represents a section with generated content."""

    title: str
    content: str
    subsections: List[str]
    level: int = 1  # Default level is 1 (main section)

    def to_dict(self) -> Dict[str, Any]:
        """Convert section to dictionary representation."""
        return {
            "title": self.title,
            "content": self.content,
            "subsections": self.subsections,
            "level": self.level,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "GeneratedSection":
        """Create a GeneratedSection from dictionary data."""
        return cls(
            title=data["title"],
            content=data["content"],
            subsections=data["subsections"],
            level=data.get("level", 1),  # Default to level 1 if not specified
        )


class DocumentParser:
    """Handles parsing, validating, and processing document structures."""

    @staticmethod
    def parse_json(json_string: str) -> Dict[str, Any]:
        """Parse a JSON string into a dictionary, with simple error handling."""
        import json
        import re

        # Clean up the JSON string to handle common issues
        def clean_json(text: str) -> str:
            # Remove any non-JSON content before the first {
            start = text.find("{")
            if start > 0:
                text = text[start:]

            # Remove any content after the last }
            end = text.rfind("}")
            if end >= 0 and end < len(text) - 1:
                text = text[: end + 1]

            # Remove trailing commas (common JSON error)
            text = re.sub(r",\s*}", "}", text)
            text = re.sub(r",\s*]", "]", text)

            return text

        try:
            # First, try to parse the string directly
            return json.loads(json_string)
        except json.JSONDecodeError:
            # If that fails, try cleaning it first
            cleaned = clean_json(json_string)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                # If still fails, give a helpful error message
                print(f"Failed to parse JSON: {e}")
                print(f"First 100 chars of cleaned JSON: {cleaned[:100]}...")
                raise ValueError(f"Could not parse JSON: {e}")

    @staticmethod
    def extract_json(text: str) -> str:
        """Extract JSON from text that might contain non-JSON content."""
        # Handle markdown code blocks
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                return text[start:end].strip()

        # Find the first { and last }
        start = text.find("{")
        end = text.rfind("}")

        if start >= 0 and end > start:
            return text[start : end + 1]

        # If no clear JSON structure, return the original text
        return text

    @staticmethod
    def validate_document_plan(data: Dict[str, Any]) -> bool:
        """Validate that the document plan has the required structure."""
        # Check for required top-level keys
        required_keys = ["introduction", "main_sections", "conclusion", "total_estimated_length"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Document plan missing required key: {key}")

        # Check that main_sections is a list
        if not isinstance(data["main_sections"], list):
            raise ValueError("main_sections must be a list")

        # Basic validation for introduction and conclusion
        for section_key in ["introduction", "conclusion"]:
            section = data[section_key]
            if not isinstance(section, dict):
                raise ValueError(f"{section_key} must be an object")
            if "title" not in section:
                raise ValueError(f"{section_key} missing title")
            if "subsections" not in section:
                raise ValueError(f"{section_key} missing subsections")

        return True

    @staticmethod
    def parse_critiques(critique_text: str) -> Dict[int, str]:
        """Parse critique text into section-specific critiques."""
        section_critiques = {}
        current_section = None
        current_critique = []

        # Find sections using regex pattern
        import re

        section_pattern = r"SECTION\s+(\d+):\s*\n(.*?)(?=SECTION\s+\d+:|$)"
        matches = re.findall(section_pattern, critique_text, re.DOTALL)

        for match in matches:
            section_num = int(match[0])
            critique = match[1].strip()
            section_critiques[section_num] = critique

        # Fallback to line-by-line parsing if regex didn't work
        if not section_critiques:
            for line in critique_text.split("\n"):
                if line.startswith("SECTION ") and ":" in line:
                    # Save the previous section critique if exists
                    if current_section is not None and current_critique:
                        section_critiques[current_section] = "\n".join(current_critique)
                        current_critique = []

                    # Extract section number
                    try:
                        section_num = int(line[8 : line.find(":")])
                        current_section = section_num
                    except ValueError:
                        continue
                elif current_section is not None:
                    current_critique.append(line)

            # Add the last section critique
            if current_section is not None and current_critique:
                section_critiques[current_section] = "\n".join(current_critique)

        return section_critiques
