#!/usr/bin/env python3
"""
Document Generator - Example Usage

This script demonstrates how to use the modular document generation system
to create well-structured documents on various topics.
"""

import os
import argparse
from dotenv import load_dotenv

# Import base components first
from core.document_generator import DocumentGenerator

def main():
    """Main function to demonstrate document generation."""
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate a document on a specified topic")
    parser.add_argument("--title", type=str, default="The Impact of Artificial Intelligence on Healthcare",
                        help="Title of the document to generate")
    parser.add_argument("--sections", type=int, default=5,
                        help="Number of main sections to generate (default: 5)")
    parser.add_argument("--pages", type=int, default=None, 
                        help="Approximate length in pages (1 page ≈ 500 words)")
    parser.add_argument("--template", type=str, default="academic",
                        choices=["academic", "report", "blog"],
                        help="Template to use for document formatting (default: academic)")
    parser.add_argument("--format", type=str, default="markdown",
                        choices=["markdown", "html", "text"],
                        help="Output format (default: markdown)")
    parser.add_argument("--output", type=str, default="generated_document.md",
                        help="Output file path (default: generated_document.md)")
    parser.add_argument("--api-key", type=str, help="Directly provide Google API key (overrides environment variable)")
    parser.add_argument("--mock", action="store_true", help="Use mock provider instead of Gemini API")
    
    args = parser.parse_args()
    
    # Determine which provider to use
    if args.mock:
        # Import the MockProvider only if needed
        from core.modules.content_generator import MockProvider
        print("Using mock provider for demonstration purposes")
        model_provider = MockProvider()
        generator = DocumentGenerator(model_provider=model_provider)
    else:
        # Get API key
        api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
        
        # Validate API key exists
        if not api_key:
            print("ERROR: No API key provided. Please set GOOGLE_API_KEY in .env file or use --api-key option.")
            print("Alternatively, use --mock to run with mock data for demonstration purposes.")
            return
            
        try:
            # Only import GeminiProvider when needed
            from core.modules.content_generator import GeminiProvider
            
            print(f"Initializing Gemini provider with API key: {api_key[:5]}...")
            model_provider = GeminiProvider(api_key=api_key, model_name="gemini-2.0-flash-thinking-exp-01-21")
            generator = DocumentGenerator(model_provider=model_provider)
        except ImportError:
            print("Error: Google Generative AI package not found.")
            print("Please install it with: pip install google-generativeai")
            print("Or use --mock to run with mock data for demonstration.")
            return
        except Exception as e:
            print(f"Error initializing Gemini API: {e}")
            print("Please check your API key and permissions.")
            print("You can use --mock flag to run with mock data for demonstration purposes.")
            return
    
    # Generate document
    print(f"Generating document: '{args.title}'")
    print(f"Using template: {args.template}")
    print(f"Output format: {args.format}")
    
    # Calculate number of sections based on pages if specified
    num_sections = args.sections
    words_per_page = 500  # Standard approximation
    
    if args.pages:
        # Calculate total words based on pages
        total_words = args.pages * words_per_page
        
        # For academic template, limit the number of sections
        if args.template == "academic":
            # Academic papers typically have 5-7 sections regardless of length
            # Introduction, 3-5 main sections, and conclusion
            if args.pages <= 10:
                num_sections = 5  # Intro, 3 main sections, conclusion
            elif args.pages <= 20:
                num_sections = 6  # Intro, 4 main sections, conclusion
            else:
                num_sections = 7  # Intro, 5 main sections, conclusion
        else:
            # For non-academic templates, use more flexible section counts
            # but still avoid excessive sections
            main_sections = min(10, max(3, args.pages // 2))  # At least 3, at most 10 main sections
            num_sections = main_sections + 2  # Add intro and conclusion
        
        print(f"Targeting approximately {args.pages} pages ({total_words} words)")
        print(f"Document will have {num_sections} sections with section-specific word counts")
    
    try:
        document = generator.generate_document(
            title=args.title,
            num_sections=num_sections,
            template_name=args.template,
            output_format=args.format,
            output_path=args.output,
            target_length_words=args.pages * words_per_page if args.pages else None
        )
        
        print(f"\nDocument Preview (first 300 characters):")
        print("-" * 50)
        preview = document[:300] + "..." if len(document) > 300 else document
        print(preview)
        
        print(f"\nDocument saved to: {args.output}")
    except Exception as e:
        print(f"Error generating document: {e}")
        print("You can use --mock flag to run with mock data for demonstration purposes.")
    
if __name__ == "__main__":
    main() 