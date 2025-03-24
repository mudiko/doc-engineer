from core.document_generator import DocumentGenerator

def main():
    # Initialize document generator
    generator = DocumentGenerator(
        model_name="gemini-2.0-flash-thinking-exp-01-21",
        use_citations=False,  # Disable citations for now
        use_search=False     # Disable search for now
    )
    
    # Generate document
    topic = "The Impact of Artificial Intelligence on Healthcare"
    target_length = 5000  # words
    
    document = generator.generate_document(
        topic=topic,
        target_length=target_length,
        output_file="generated_document.md"
    )
    
    print("\nDocument generation complete!")

if __name__ == "__main__":
    main() 