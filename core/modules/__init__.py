"""
Document Generator Modules

This package contains modular components for document generation:
1. Document Parsing - Handles parsing and structural representation of documents
2. Content Generation - Manages AI-based content generation
3. Template Handling - Provides document templates and formatting structures
4. Output Formatting - Handles document formatting and export to different formats
"""

# Document Parser Module
from .document_parser import (
    Section,
    DocumentPlan,
    GeneratedSection,
    DocumentParser
)

# Content Generator Module
from .content_generator import (
    ModelProvider,
    GeminiProvider,
    ContentGenerator
)

# Template Handler Module
from .template_handler import (
    Template,
    TemplateHandler,
    TemplateStrategy,
    AcademicTemplateStrategy,
    ReportTemplateStrategy,
    TemplateFactory
)

# Output Formatter Module
from .output_formatter import (
    OutputFormatter,
    MarkdownFormatter,
    HTMLFormatter,
    TextFormatter,
    FormatterFactory,
    DocumentReviewFormatter
)

__all__ = [
    # Document Parser
    'Section', 'DocumentPlan', 'GeneratedSection', 'DocumentParser',
    
    # Content Generator
    'ModelProvider', 'GeminiProvider', 'ContentGenerator',
    
    # Template Handler
    'Template', 'TemplateHandler', 'TemplateStrategy',
    'AcademicTemplateStrategy', 'ReportTemplateStrategy', 'TemplateFactory',
    
    # Output Formatter
    'OutputFormatter', 'MarkdownFormatter', 'HTMLFormatter', 'TextFormatter',
    'FormatterFactory', 'DocumentReviewFormatter'
] 