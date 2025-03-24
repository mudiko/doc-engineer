import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def generate_content(topic: str) -> str:
    """Generate content about a topic using Gemini."""
    # Configure Gemini API
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.0-flash-thinking-exp-01-21")
    
    # Create prompt
    prompt = f"""Write a comprehensive academic article about {topic}.

The article should:
1. Start with an introduction explaining the topic and its significance
2. Include 3-4 main sections covering key aspects
3. Use academic language and tone
4. Be well-structured with clear paragraphs
5. End with a conclusion

Please write in a clear, professional style suitable for academic publication."""

    try:
        # Generate content
        response = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.7,
                "max_output_tokens": 2000,
            }
        )
        
        return response.text
        
    except Exception as e:
        print(f"Error generating content: {e}")
        return "Error generating content. Please try again."

def main():
    # Example topic
    topic = "The Impact of Artificial Intelligence on Healthcare"
    
    print(f"Generating content about: {topic}")
    print("-" * 50)
    
    content = generate_content(topic)
    print(content)

if __name__ == "__main__":
    main() 