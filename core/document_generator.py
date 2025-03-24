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
        print("[1/5] Creating document plan...")
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
        print(f"[2/5] Using '{template_name}' template")
        template = get_template(template_name)
        
        # Step 3: Generate content for each section
        print("[3/5] Generating content...")
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
        
        # Step 4: Generate document-wide critique
        print("[4/5] Evaluating document coherence and quality...")
        
        # First check for consistency issues between sections
        consistency_issues = []
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
        
        # Generate critiques for the entire document
        print("Generating document-wide critique...")
        section_critiques = self.content_generator.evaluate_document_sections(
            title, 
            generated_sections
        )
        
        if section_critiques:
            print(f"Generated critiques for {len(section_critiques)} sections")
            
        # Step 5: Revise and improve each section based on critiques
        print("[5/5] Improving sections based on critique...")
        
        # Only revise if we have meaningful critiques
        if section_critiques or consistency_issues:
            improved_sections = []
            
            for i, section in enumerate(generated_sections):
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
                
                # Only revise if we have critique feedback
                if combined_critique.strip():
                    print(f"  • Improving {section.title} based on critique")
                    # Get the section from the document plan
                    plan_section = document_plan.sections[i]
                    
                    # Get context from previous sections
                    previous_context = [s.title + "\n" + s.content[:300] for s in improved_sections]
                    
                    # Revise the section based on critique
                    improved_section = self.content_generator.revise_section(
                        section,
                        combined_critique,
                        plan_section,
                        previous_context
                    )
                    improved_sections.append(improved_section)
                else:
                    # No critique, just keep the original
                    improved_sections.append(section)
            
            # Replace the sections with improved versions
            generated_sections = improved_sections
            print(f"Improved {len(section_critiques) + len(consistency_issues)} sections based on critique")
        else:
            print("No issues found - document is already well-structured and coherent")
        
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