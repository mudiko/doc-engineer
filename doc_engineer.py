#!/usr/bin/env python3
"""
Doc Engineer - A powerful single-shot document generation system

This module provides the command-line interface for the Doc Engineer system.
"""

import os
import sys
import argparse
from dotenv import load_dotenv

from core.document_generator import DocumentGenerator


def main():
    """Command-line interface for Doc Engineer."""
    # Load environment variables from .env file if present
    load_dotenv()

    # Set up command-line argument parser
    parser = argparse.ArgumentParser(
        description="Doc Engineer - Generate well-structured documents on any topic"
    )
    parser.add_argument(
        "title", 
        nargs="?", 
        default="Advancements of AI", 
        help="Title of the document to generate"
    )
    parser.add_argument(
        "--sections", 
        type=int, 
        default=5, 
        help="Number of main sections to generate"
    )
    parser.add_argument(
        "--pages",
        type=int,
        default=5,
        help="Approximate length in pages (1 page ≈ 500 words)",
    )
    parser.add_argument(
        "--template",
        choices=["academic", "report", "blog"],
        default="academic",
        help="Template to use for document formatting",
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "html", "text"],
        default="markdown",
        help="Output format",
    )
    parser.add_argument(
        "--output", 
        default="generated_document.md", 
        help="Output file path"
    )
    parser.add_argument(
        "--api-key", 
        default=None, 
        help="Google API key (overrides environment variable)"
    )
    parser.add_argument(
        "--mock", 
        action="store_true", 
        help="Use mock provider instead of Gemini API"
    )
    parser.add_argument(
        "--hide-tokens", 
        action="store_true", 
        help="Hide detailed token usage statistics"
    )
    parser.add_argument(
        "--with-citations", 
        action="store_true", 
        help="Include citations from academic papers"
    )
    parser.add_argument(
        "--scopus-api-key", 
        default=None, 
        help="Scopus API key for citation search (overrides environment variable)"
    )
    parser.add_argument(
        "--ieee-api-key", 
        default=None, 
        help="IEEE API key for citation search (overrides environment variable)"
    )
    parser.add_argument(
        "--use-semantic-scholar",
        action="store_true",
        help="Use Semantic Scholar for citation retrieval (default)"
    )
    parser.add_argument(
        "--use-findpapers",
        action="store_true",
        help="Use findpapers instead of Semantic Scholar for citation retrieval"
    )

    args = parser.parse_args()

    # Check if we're using environment variable for API key
    api_key = args.api_key or os.getenv("GOOGLE_API_KEY")
    
    # Set environment variables for citation APIs if provided
    if args.scopus_api_key:
        os.environ["SCOPUS_API_TOKEN"] = args.scopus_api_key
    if args.ieee_api_key:
        os.environ["IEEE_API_TOKEN"] = args.ieee_api_key

    # Determine which citation source to use
    # By default, use Semantic Scholar unless findpapers is explicitly requested
    use_semantic_scholar = not args.use_findpapers
    
    if args.use_findpapers:
        print("Using findpapers for citations as requested")
    else:
        print("Using Semantic Scholar for citations (default)")
        
    # Set up document generator
    generator = DocumentGenerator(
        api_key=api_key, 
        mock=args.mock,
        use_semantic_scholar=use_semantic_scholar
    )

    # Print document generation info
    print("\nDoc Engineer - Generating document: '{}'".format(args.title))
    print("Using template: {}".format(args.template))
    print("Output format: {}".format(args.format))

    # Generate document with parameters
    document = generator.generate_document(
        title=args.title,
        num_sections=args.sections,
        template_name=args.template,
        output_format=args.format,
        output_path=args.output,
        target_length_words=args.pages * 500,  # Convert pages to approximate word count
        show_tokens=not args.hide_tokens,
        with_citations=args.with_citations,
    )

    print("\nDocument successfully generated!")
    print("Saved to: {}".format(args.output))


if __name__ == "__main__":
    main()
