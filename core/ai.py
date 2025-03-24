from typing import List, Dict, Any
from dataclasses import dataclass
import os
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@dataclass
class Source:
    authors: List[str]
    year: int
    title: str
    venue: str
    doi: str
    content: str
    credibility_score: float = 0.0

@dataclass
class SynthesisConfig:
    style_guide: Dict[str, Any]
    tone: str = "academic"
    field_terminology: bool = True
    citation_integration: bool = True

class AI:
    def __init__(self, model_name: str = "gemini-pro"):
        self.model_name = model_name
        # Configure Gemini API
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        self.synthesis_config = SynthesisConfig(
            style_guide={
                "tone": "academic",
                "citation_style": "APA",
                "paragraph_structure": "topic-sentence",
                "coherence": "high"
            }
        )
    
    def synthesize(
        self,
        topic: str,
        sources: List[Source],
        style_guide: Dict[str, Any] = None
    ) -> str:
        """
        Synthesize content from multiple sources while maintaining academic rigor
        and proper citation integration.
        """
        if style_guide:
            self.synthesis_config.style_guide.update(style_guide)
            
        # Prepare source content for synthesis
        source_texts = [source.content for source in sources]
        
        # Generate synthesis prompt
        prompt = self._create_synthesis_prompt(topic, source_texts)
        
        # Generate content using Gemini API
        try:
            response = self.model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 2000,
                }
            )
            
            synthesized_content = response.text
            return synthesized_content
            
        except Exception as e:
            print(f"Error generating content: {e}")
            return "Error generating content. Please try again."
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for content generation."""
        return f"""You are an expert academic writer specializing in research synthesis.
Your task is to generate well-structured, academically rigorous content that:
1. Maintains a {self.synthesis_config.tone} tone
2. Uses appropriate field-specific terminology
3. Integrates citations naturally
4. Follows {self.synthesis_config.style_guide['citation_style']} citation style
5. Maintains high coherence and logical flow
6. Uses topic-sentence paragraph structure

Please ensure your output is:
- Well-organized with clear sections
- Free of plagiarism
- Properly cited
- Academically rigorous
- Easy to follow"""
    
    def _create_synthesis_prompt(self, topic: str, source_texts: List[str]) -> str:
        """Create a prompt for content synthesis."""
        return f"""Topic: {topic}

Sources:
{self._format_sources(source_texts)}

Please generate a comprehensive synthesis that:
1. Introduces the topic and its significance
2. Synthesizes key findings from the sources
3. Identifies patterns and relationships
4. Addresses any contradictions or gaps
5. Concludes with implications and future directions

Requirements:
- Use academic language and tone
- Integrate citations naturally
- Maintain logical flow between ideas
- Use field-specific terminology appropriately
- Structure paragraphs with clear topic sentences
- Ensure coherence between sections"""
    
    def _format_sources(self, source_texts: List[str]) -> str:
        """Format source texts for prompt creation."""
        return "\n\n".join(f"Source {i+1}:\n{text}" for i, text in enumerate(source_texts)) 