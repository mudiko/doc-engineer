"""
Document Generator

This module provides the core document generation functionality.
"""

import os
from typing import Dict, List, Optional, Any, Tuple

from .modules.content_generator import ContentGenerator, GeminiProvider, ModelProvider
from .modules.document_parser import Section, DocumentPlan, GeneratedSection, DocumentParser
from .modules.templates import get_template, Template


class DocumentGenerator:
    """Class for generating complete documents with AI assistance."""

    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-thinking-exp-01-21", model_provider: Optional[ModelProvider] = None):
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
        print(f"Document generator initialized successfully")

    def generate_document(self, title: str, num_sections: int = 5, 
                         template_name: str = "academic", 
                         output_format: str = "markdown",
                         output_path: Optional[str] = None,
                         target_length_words: Optional[int] = None) -> str:
        """
        Generate a complete document based on the given title.
        
        Args:
            title (str): The title of the document
            num_sections (int): The number of sections to generate (default: 5)
            template_name (str): The template to use (default: "academic")
            output_format (str): Format to output the document in (default: "markdown")
            output_path (Optional[str]): Path to save the document (default: None)
            target_length_words (Optional[int]): Target document length in words (default: None)
            
        Returns:
            str: The generated document content
        """
        print("=== Generating Document ===")
        
        # Step 1: Create a document plan
        print("[1/4] Creating document plan...")
        document_plan = self.content_generator.create_document_plan(
            title, 
            num_sections,
            target_length_words=target_length_words
        )
        print(f"Plan created with {len(document_plan.sections)} sections")
        
        # Display estimated length information if target length was specified
        if target_length_words:
            print(f"Target length: {target_length_words} words (≈{target_length_words/500:.1f} pages)")
            print(f"Planned length: {document_plan.total_estimated_length} words")
            
            # Show section breakdown
            print(f"Section breakdown:")
            print(f"  - Introduction: {document_plan.introduction.estimated_length} words")
            for i, section in enumerate(document_plan.main_sections):
                print(f"  - {section.title}: {section.estimated_length} words")
            print(f"  - Conclusion: {document_plan.conclusion.estimated_length} words")
        
        # Step 2: Apply template
        print(f"[2/4] Using '{template_name}' template")
        template = get_template(template_name)
        
        # Step 3: Generate content for each section
        print("[3/4] Generating content...")
        generated_sections = []
        
        for i, section in enumerate(document_plan.sections):
            print(f"  • {i+1}/{len(document_plan.sections)}: {section.title}")
            
            # Generate content for this section
            gen_section = self.content_generator.generate_section_content(
                title, 
                section, 
                generated_sections.copy()
            )
            
            generated_sections.append(gen_section)
        
        # Step 4: Check consistency and evaluate
        print("[4/4] Checking consistency and evaluating...")
        consistency_issues = []
        
        # Check consistency between sections
        for i, section in enumerate(generated_sections[1:], 1):
            previous_sections = generated_sections[:i]
            consistency_report = self.content_generator.check_consistency(
                section.content, 
                section.title, 
                previous_sections
            )
            if consistency_report:
                consistency_issues.append((section.title, consistency_report))
        
        if consistency_issues:
            print(f"Found {len(consistency_issues)} consistency issues")
        else:
            print("No consistency issues found")
        
        # Evaluate document sections
        section_critiques = self.content_generator.evaluate_document_sections(
            title, 
            generated_sections
        )
        
        # Format the document
        print(f"Formatting with {template_name} template...")
        formatted_document = template.format_document(
            title=title,
            sections=generated_sections,
            output_format=output_format
        )
        
        # Save document if output path is provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(formatted_document)
            print(f"Document saved to {output_path}")
        
        print(f"Generated {len(generated_sections)} sections with ~{sum(len(s.content.split()) for s in generated_sections)} words")
        print("=== Done ===")
            
        return formatted_document 