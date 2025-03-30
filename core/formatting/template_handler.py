from typing import Dict, Any, List, Optional
import os
import json
from dataclasses import dataclass
from pathlib import Path
from abc import ABC, abstractmethod

# Updated import after refactoring
from core.planning.document_parser import Section, DocumentPlan, GeneratedSection


@dataclass
class Template:
    """Document template structure."""

    name: str
    description: str
    sections: List[str]
    formatting: Dict[str, Any]
    metadata: Dict[str, Any]


class TemplateStrategy(ABC):
    """Abstract base class for template strategies."""

    @abstractmethod
    def apply_template(self, sections: List[GeneratedSection], template: Template) -> str:
        """Apply a template to generated sections."""
        pass

    def apply_to_section(self, section: GeneratedSection) -> GeneratedSection:
        """Apply formatting to an individual section."""
        # Default implementation just returns the section unchanged
        return section


class TemplateHandler:
    """Handles document templates and provides template functionality."""

    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = templates_dir
        self.templates = {}
        self._load_templates()

    def _load_templates(self) -> None:
        """Load templates from the templates directory."""
        templates_path = Path(self.templates_dir)
        if not templates_path.exists():
            os.makedirs(templates_path, exist_ok=True)
            self._create_default_templates()

        for template_file in templates_path.glob("*.json"):
            try:
                with open(template_file, "r", encoding="utf-8") as f:
                    template_data = json.load(f)
                    template = Template(
                        name=template_data.get("name", template_file.stem),
                        description=template_data.get("description", ""),
                        sections=template_data.get("sections", []),
                        formatting=template_data.get("formatting", {}),
                        metadata=template_data.get("metadata", {}),
                    )
                    self.templates[template.name] = template
            except Exception as e:
                print(f"Error loading template {template_file}: {e}")

    def _create_default_templates(self) -> None:
        """Create default templates if none exist."""
        default_templates = [
            {
                "name": "academic",
                "description": "Academic article template with standard sections",
                "sections": [
                    "Introduction",
                    "Literature Review",
                    "Methodology",
                    "Results",
                    "Discussion",
                    "Conclusion",
                ],
                "formatting": {
                    "heading_style": "atx",  # # Heading 1, ## Heading 2, etc.
                    "citation_style": "APA",
                    "include_toc": True,
                    "include_abstract": True,
                    "include_references": True,
                },
                "metadata": {
                    "font": "Times New Roman",
                    "font_size": 12,
                    "line_spacing": 2.0,
                    "margins": "1in",
                },
            },
            {
                "name": "report",
                "description": "Business report template",
                "sections": [
                    "Executive Summary",
                    "Introduction",
                    "Findings",
                    "Recommendations",
                    "Conclusion",
                ],
                "formatting": {
                    "heading_style": "atx",
                    "citation_style": "Chicago",
                    "include_toc": True,
                    "include_abstract": False,
                    "include_references": True,
                },
                "metadata": {
                    "font": "Arial",
                    "font_size": 11,
                    "line_spacing": 1.5,
                    "margins": "1in",
                },
            },
        ]

        templates_path = Path(self.templates_dir)
        for template in default_templates:
            template_path = templates_path / f"{template['name']}.json"
            with open(template_path, "w", encoding="utf-8") as f:
                json.dump(template, f, indent=2)
            self.templates[template["name"]] = Template(
                name=template["name"],
                description=template["description"],
                sections=template["sections"],
                formatting=template["formatting"],
                metadata=template["metadata"],
            )

    def get_template(self, name: str) -> Optional[Template]:
        """Get a template by name."""
        return self.templates.get(name)

    def list_templates(self) -> List[str]:
        """List available templates."""
        return list(self.templates.keys())

    def create_template(self, template: Template) -> None:
        """Create a new template."""
        template_path = Path(self.templates_dir) / f"{template.name}.json"
        with open(template_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "name": template.name,
                    "description": template.description,
                    "sections": template.sections,
                    "formatting": template.formatting,
                    "metadata": template.metadata,
                },
                f,
                indent=2,
            )
        self.templates[template.name] = template

    def get_template_strategy(self, template_name: str) -> TemplateStrategy:
        """Get the appropriate template strategy for a template name."""
        return TemplateFactory.create_strategy(template_name)

    def apply_template_to_plan(self, plan: DocumentPlan, template_name: str) -> DocumentPlan:
        """Apply a template to a document plan."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        # Adjust sections based on template if needed
        # This is a simplistic implementation - in a real system, this would be more sophisticated
        # to match template sections with plan sections
        return plan

    def get_formatting_options(self, template_name: str) -> Dict[str, Any]:
        """Get formatting options for a template."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        return template.formatting

    def get_metadata(self, template_name: str) -> Dict[str, Any]:
        """Get metadata for a template."""
        template = self.get_template(template_name)
        if not template:
            raise ValueError(f"Template not found: {template_name}")

        return template.metadata


class AcademicTemplateStrategy(TemplateStrategy):
    """Strategy for academic document templates."""

    def apply_template(self, sections: List[GeneratedSection], template: Template) -> str:
        """Apply an academic template to generated sections."""
        document = ""

        # Add title and abstract if needed
        if template.formatting.get("include_abstract", False):
            document += "# Abstract\n\n"
            document += "Abstract content would be here.\n\n"

        # Add table of contents if needed
        if template.formatting.get("include_toc", False):
            document += "# Table of Contents\n\n"
            document += "Table of contents would be generated here.\n\n"

        # Add sections
        for section in sections:
            document += f"# {section.title}\n\n"
            document += f"{section.content}\n\n"

        # Add references if needed
        if template.formatting.get("include_references", False):
            document += "# References\n\n"
            document += "References would be added here.\n\n"

        return document

    def apply_to_section(self, section: GeneratedSection) -> GeneratedSection:
        """Apply academic formatting to a section."""
        # For academic sections, we may want to add specific formatting
        # In this simple implementation, we just add a level attribute if missing
        if not hasattr(section, "level"):
            # Set section level (1 for main sections)
            section_dict = section.__dict__.copy()
            section_dict["level"] = 1
            return GeneratedSection(**section_dict)
        return section


class ReportTemplateStrategy(TemplateStrategy):
    """Strategy for business report templates."""

    def apply_template(self, sections: List[GeneratedSection], template: Template) -> str:
        """Apply a report template to generated sections."""
        document = ""

        # Add title page
        document += "# " + template.name.upper() + "\n\n"
        document += "Prepared by: Author Name\n\n"
        document += "Date: [Current Date]\n\n"

        # Add executive summary if needed
        if "Executive Summary" in template.sections:
            document += "# Executive Summary\n\n"
            # This would typically be generated separately
            document += "Executive summary would be here.\n\n"

        # Add sections
        for section in sections:
            document += f"# {section.title}\n\n"
            document += f"{section.content}\n\n"

        # Add appendices if needed
        if template.formatting.get("include_appendices", False):
            document += "# Appendices\n\n"
            document += "Appendices would be added here.\n\n"

        return document

    def apply_to_section(self, section: GeneratedSection) -> GeneratedSection:
        """Apply report formatting to a section."""
        # For report sections, we may want specific formatting
        # In this simple implementation, we just ensure a level attribute exists
        if not hasattr(section, "level"):
            # Set section level (1 for main sections)
            section_dict = section.__dict__.copy()
            section_dict["level"] = 1
            return GeneratedSection(**section_dict)
        return section


class TemplateFactory:
    """Factory for creating template strategies."""

    @staticmethod
    def create_strategy(template_name: str) -> TemplateStrategy:
        """Create a template strategy based on template name."""
        if template_name == "academic":
            return AcademicTemplateStrategy()
        elif template_name == "report":
            return ReportTemplateStrategy()
        else:
            # Default to academic
            return AcademicTemplateStrategy()
