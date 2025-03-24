"""
Templates module for the document generator.

This module provides templates for formatting documents in different styles.
"""

from typing import Dict, Any, List
from .document_parser import GeneratedSection, DocumentPlan


class Template:
    """Base class for document templates."""
    
    @staticmethod
    def format_document(
        title: str, 
        sections: List[GeneratedSection], 
        output_format: str = "markdown"
    ) -> str:
        """
        Format the document according to the template.
        
        Args:
            title: Document title
            sections: List of generated sections
            output_format: Output format (markdown, html, text)
            
        Returns:
            Formatted document as a string
        """
        # This should be implemented by subclasses
        raise NotImplementedError("Subclasses must implement format_document")


class AcademicTemplate(Template):
    """Academic paper template with formal structure."""
    
    @staticmethod
    def format_document(
        title: str, 
        sections: List[GeneratedSection], 
        output_format: str = "markdown"
    ) -> str:
        """Format document as an academic paper."""
        if output_format == "markdown":
            document = f"# {title}\n\n"
            
            # Add abstract placeholder
            document += "## Abstract\n\n"
            document += "_This paper explores " + title.lower() + "._\n\n"
            
            # Add each section
            for section in sections:
                # Determine heading level based on section level (default to 2)
                level = section.level if hasattr(section, 'level') else 2
                heading_prefix = '#' * level
                
                document += f"{heading_prefix} {section.title}\n\n"
                document += f"{section.content}\n\n"
            
            # Add references section
            document += "## References\n\n"
            document += "* References will be generated based on citations in the text.\n\n"
            
            return document
            
        elif output_format == "html":
            # Simple HTML template
            document = f"<!DOCTYPE html>\n<html>\n<head>\n<title>{title}</title>\n</head>\n<body>\n"
            document += f"<h1>{title}</h1>\n"
            
            # Add abstract
            document += "<h2>Abstract</h2>\n"
            document += f"<p><em>This paper explores {title.lower()}.</em></p>\n"
            
            # Add each section
            for section in sections:
                level = section.level if hasattr(section, 'level') else 2
                document += f"<h{level}>{section.title}</h{level}>\n"
                
                # Convert basic markdown to HTML (paragraphs)
                paragraphs = section.content.split("\n\n")
                for paragraph in paragraphs:
                    if paragraph.strip():
                        document += f"<p>{paragraph}</p>\n"
            
            # Add references
            document += "<h2>References</h2>\n"
            document += "<ul><li>References will be generated based on citations in the text.</li></ul>\n"
            
            document += "</body>\n</html>"
            return document
            
        else:  # Plain text
            document = f"{title.upper()}\n{'=' * len(title)}\n\n"
            
            # Add abstract
            document += "ABSTRACT\n--------\n\n"
            document += f"This paper explores {title.lower()}.\n\n"
            
            # Add each section
            for section in sections:
                document += f"{section.title.upper()}\n{'-' * len(section.title)}\n\n"
                document += f"{section.content}\n\n"
            
            # Add references
            document += "REFERENCES\n----------\n\n"
            document += "* References will be generated based on citations in the text.\n\n"
            
            return document


class ReportTemplate(Template):
    """Business report template with executive summary."""
    
    @staticmethod
    def format_document(
        title: str, 
        sections: List[GeneratedSection], 
        output_format: str = "markdown"
    ) -> str:
        """Format document as a business report."""
        if output_format == "markdown":
            document = f"# {title}\n\n"
            
            # Add executive summary
            document += "## Executive Summary\n\n"
            document += "_This report provides key insights about " + title.lower() + "._\n\n"
            
            # Add each section
            for section in sections:
                level = section.level if hasattr(section, 'level') else 2
                heading_prefix = '#' * level
                
                document += f"{heading_prefix} {section.title}\n\n"
                document += f"{section.content}\n\n"
            
            # Add appendices
            document += "## Appendices\n\n"
            document += "* Supplementary materials and references\n\n"
            
            return document
            
        elif output_format == "html":
            # HTML template for report
            document = f"<!DOCTYPE html>\n<html>\n<head>\n<title>{title}</title>\n</head>\n<body>\n"
            document += f"<h1>{title}</h1>\n"
            
            # Add executive summary
            document += "<h2>Executive Summary</h2>\n"
            document += f"<p><em>This report provides key insights about {title.lower()}.</em></p>\n"
            
            # Add each section
            for section in sections:
                level = section.level if hasattr(section, 'level') else 2
                document += f"<h{level}>{section.title}</h{level}>\n"
                
                # Convert basic markdown to HTML
                paragraphs = section.content.split("\n\n")
                for paragraph in paragraphs:
                    if paragraph.strip():
                        document += f"<p>{paragraph}</p>\n"
            
            # Add appendices
            document += "<h2>Appendices</h2>\n"
            document += "<ul><li>Supplementary materials and references</li></ul>\n"
            
            document += "</body>\n</html>"
            return document
            
        else:  # Plain text
            document = f"{title.upper()}\n{'=' * len(title)}\n\n"
            
            # Add executive summary
            document += "EXECUTIVE SUMMARY\n-----------------\n\n"
            document += f"This report provides key insights about {title.lower()}.\n\n"
            
            # Add each section
            for section in sections:
                document += f"{section.title.upper()}\n{'-' * len(section.title)}\n\n"
                document += f"{section.content}\n\n"
            
            # Add appendices
            document += "APPENDICES\n----------\n\n"
            document += "* Supplementary materials and references\n\n"
            
            return document


class BlogTemplate(Template):
    """Blog post template with informal style."""
    
    @staticmethod
    def format_document(
        title: str, 
        sections: List[GeneratedSection], 
        output_format: str = "markdown"
    ) -> str:
        """Format document as a blog post."""
        if output_format == "markdown":
            document = f"# {title}\n\n"
            
            # Get the first section (introduction)
            if sections and sections[0].title.lower() == "introduction":
                document += sections[0].content + "\n\n"
                sections = sections[1:]  # Remove introduction from sections list
            
            # Add table of contents
            document += "## Table of Contents\n\n"
            for i, section in enumerate(sections, 1):
                # Skip conclusion for TOC
                if section.title.lower() != "conclusion":
                    document += f"{i}. [{section.title}](#{section.title.lower().replace(' ', '-')})\n"
            document += "\n\n"
            
            # Add each section
            for section in sections:
                document += f"## {section.title}\n\n"
                document += f"{section.content}\n\n"
            
            return document
            
        elif output_format == "html":
            # HTML template for blog
            document = f"<!DOCTYPE html>\n<html>\n<head>\n<title>{title}</title>\n</head>\n<body>\n"
            document += f"<h1>{title}</h1>\n"
            
            # Get the first section (introduction)
            if sections and sections[0].title.lower() == "introduction":
                paragraphs = sections[0].content.split("\n\n")
                for paragraph in paragraphs:
                    if paragraph.strip():
                        document += f"<p>{paragraph}</p>\n"
                sections = sections[1:]  # Remove introduction
            
            # Add table of contents
            document += "<h2>Table of Contents</h2>\n<ol>\n"
            for i, section in enumerate(sections, 1):
                # Skip conclusion for TOC
                if section.title.lower() != "conclusion":
                    document += f'<li><a href="#{section.title.lower().replace(" ", "-")}">{section.title}</a></li>\n'
            document += "</ol>\n"
            
            # Add each section
            for section in sections:
                section_id = section.title.lower().replace(" ", "-")
                document += f'<h2 id="{section_id}">{section.title}</h2>\n'
                
                # Convert basic markdown to HTML
                paragraphs = section.content.split("\n\n")
                for paragraph in paragraphs:
                    if paragraph.strip():
                        document += f"<p>{paragraph}</p>\n"
            
            document += "</body>\n</html>"
            return document
            
        else:  # Plain text
            document = f"{title.upper()}\n{'=' * len(title)}\n\n"
            
            # Get the first section (introduction)
            if sections and sections[0].title.lower() == "introduction":
                document += sections[0].content + "\n\n"
                sections = sections[1:]  # Remove introduction
            
            # Add table of contents
            document += "TABLE OF CONTENTS\n----------------\n\n"
            for i, section in enumerate(sections, 1):
                # Skip conclusion for TOC
                if section.title.lower() != "conclusion":
                    document += f"{i}. {section.title}\n"
            document += "\n\n"
            
            # Add each section
            for section in sections:
                document += f"{section.title.upper()}\n{'-' * len(section.title)}\n\n"
                document += f"{section.content}\n\n"
            
            return document


def get_template(template_name: str) -> Template:
    """Get a template by name."""
    templates = {
        "academic": AcademicTemplate,
        "report": ReportTemplate,
        "blog": BlogTemplate
    }
    
    if template_name.lower() not in templates:
        print(f"Warning: Template '{template_name}' not found. Using academic template instead.")
        return AcademicTemplate
        
    return templates[template_name.lower()] 