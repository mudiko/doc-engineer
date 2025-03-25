from typing import Dict, Any, List, Optional, Protocol
import os
from abc import ABC, abstractmethod
from pathlib import Path
import re
import datetime

from .document_parser import GeneratedSection


class OutputFormatter(ABC):
    """Abstract base class for output formatters."""

    @abstractmethod
    def format_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Format the document content."""
        pass

    @abstractmethod
    def save_document(self, content: str, output_path: str, metadata: Dict[str, Any] = None) -> str:
        """Save the document to a file."""
        pass


class MarkdownFormatter(OutputFormatter):
    """Formatter for Markdown output."""

    def format_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Format the document content in Markdown."""
        metadata = metadata or {}
        formatted_content = content

        # Add title if provided
        if title := metadata.get("title"):
            formatted_content = f"# {title}\n\n{formatted_content}"

        # Add author if provided
        if author := metadata.get("author"):
            # Find position after title to insert author
            if "# " in formatted_content and "\n" in formatted_content:
                title_end = formatted_content.find("\n", formatted_content.find("# "))
                formatted_content = (
                    formatted_content[: title_end + 1]
                    + f"\n*by {author}*\n\n"
                    + formatted_content[title_end + 1 :]
                )
            else:
                formatted_content = f"*by {author}*\n\n{formatted_content}"

        # Add date if needed
        if metadata.get("include_date", False):
            date_str = metadata.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))
            # Find position to insert date (after title and author)
            if "*by " in formatted_content and "\n" in formatted_content:
                author_end = formatted_content.find("\n", formatted_content.find("*by "))
                formatted_content = (
                    formatted_content[: author_end + 1]
                    + f"*{date_str}*\n\n"
                    + formatted_content[author_end + 1 :]
                )
            else:
                # Insert after title if there's no author
                if "# " in formatted_content and "\n" in formatted_content:
                    title_end = formatted_content.find("\n", formatted_content.find("# "))
                    formatted_content = (
                        formatted_content[: title_end + 1]
                        + f"\n*{date_str}*\n\n"
                        + formatted_content[title_end + 1 :]
                    )
                else:
                    formatted_content = f"*{date_str}*\n\n{formatted_content}"

        return formatted_content

    def save_document(self, content: str, output_path: str, metadata: Dict[str, Any] = None) -> str:
        """Save the document as a Markdown file."""
        # Format the content
        formatted_content = self.format_document(content, metadata)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Save the file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_content)

        return output_path


class HTMLFormatter(OutputFormatter):
    """Formatter for HTML output."""

    def format_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Format the document content as HTML."""
        metadata = metadata or {}

        # Convert Markdown headings to HTML
        html_content = self._md_to_html(content)

        # Create HTML document structure
        html = "<!DOCTYPE html>\n<html>\n<head>\n"
        html += f"<title>{metadata.get('title', 'Document')}</title>\n"

        # Add style
        html += "<style>\n"
        html += "body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 800px; margin: 0 auto; padding: 20px; }\n"
        html += "h1 { color: #333; }\n"
        html += "h2 { color: #444; margin-top: 30px; }\n"
        html += "h3 { color: #555; }\n"
        html += "p { margin-bottom: 16px; }\n"
        html += "</style>\n"

        html += "</head>\n<body>\n"

        # Add title
        if title := metadata.get("title"):
            html += f"<h1>{title}</h1>\n"

        # Add author if provided
        if author := metadata.get("author"):
            html += f"<p><em>by {author}</em></p>\n"

        # Add date if needed
        if metadata.get("include_date", False):
            date_str = metadata.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))
            html += f"<p><em>{date_str}</em></p>\n"

        # Add content
        html += html_content

        # Close HTML document
        html += "\n</body>\n</html>"

        return html

    def save_document(self, content: str, output_path: str, metadata: Dict[str, Any] = None) -> str:
        """Save the document as an HTML file."""
        # Format the content
        formatted_content = self.format_document(content, metadata)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # If output path doesn't end with .html, add it
        if not output_path.lower().endswith(".html"):
            output_path += ".html"

        # Save the file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_content)

        return output_path

    def _md_to_html(self, markdown: str) -> str:
        """Convert basic Markdown to HTML."""
        html = markdown

        # Convert headings
        html = re.sub(r"# (.*?)$", r"<h1>\1</h1>", html, flags=re.MULTILINE)
        html = re.sub(r"## (.*?)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
        html = re.sub(r"### (.*?)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)

        # Convert paragraphs (very simple approach)
        html = re.sub(r"\n\n(.*?)\n\n", r"\n<p>\1</p>\n", html)

        # Convert bold
        html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)

        # Convert italic
        html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)

        return html


class TextFormatter(OutputFormatter):
    """Formatter for plain text output."""

    def format_document(self, content: str, metadata: Dict[str, Any] = None) -> str:
        """Format the document content as plain text."""
        metadata = metadata or {}

        # Strip all markdown formatting
        text_content = self._strip_markdown(content)

        # Add title if provided
        if title := metadata.get("title"):
            text_content = f"{title.upper()}\n\n{text_content}"

        # Add author if provided
        if author := metadata.get("author"):
            text_content = f"{text_content}\nby {author}\n"

        # Add date if needed
        if metadata.get("include_date", False):
            date_str = metadata.get("date", datetime.datetime.now().strftime("%Y-%m-%d"))
            text_content = f"{text_content}\n{date_str}\n"

        return text_content

    def save_document(self, content: str, output_path: str, metadata: Dict[str, Any] = None) -> str:
        """Save the document as a plain text file."""
        # Format the content
        formatted_content = self.format_document(content, metadata)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # If output path doesn't end with .txt, add it
        if not output_path.lower().endswith(".txt"):
            output_path += ".txt"

        # Save the file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(formatted_content)

        return output_path

    def _strip_markdown(self, markdown: str) -> str:
        """Strip markdown formatting."""
        text = markdown

        # Remove headings
        text = re.sub(r"#+ (.*?)$", r"\1", text, flags=re.MULTILINE)

        # Remove bold
        text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)

        # Remove italic
        text = re.sub(r"\*(.*?)\*", r"\1", text)

        return text


class FormatterFactory:
    """Factory for creating output formatters."""

    @staticmethod
    def create_formatter(format_type: str) -> OutputFormatter:
        """Create a formatter based on format type."""
        format_type = format_type.lower()
        if format_type == "markdown" or format_type == "md":
            return MarkdownFormatter()
        elif format_type == "html":
            return HTMLFormatter()
        elif format_type == "text" or format_type == "txt":
            return TextFormatter()
        else:
            # Default to Markdown
            return MarkdownFormatter()

    @staticmethod
    def get_formatter(format_type: str) -> OutputFormatter:
        """Get a formatter based on format type (alias for create_formatter)."""
        return FormatterFactory.create_formatter(format_type)


class DocumentReviewFormatter:
    """Handles formatting document reviews and feedback."""

    @staticmethod
    def format_section_critiques(
        critiques: Dict[int, str], sections: List[GeneratedSection]
    ) -> str:
        """Format section critiques into a readable format."""
        if not critiques:
            return "No critiques available."

        review = "# Document Review\n\n"

        for section_idx, critique in critiques.items():
            if section_idx < len(sections):
                section = sections[section_idx]
                review += f"## {section.title}\n\n"
                review += critique + "\n\n"
            else:
                review += f"## Section {section_idx}\n\n"
                review += critique + "\n\n"

        return review

    @staticmethod
    def format_consistency_report(report: str) -> str:
        """Format consistency report into a readable format."""
        return "# Document Consistency Report\n\n" + report
