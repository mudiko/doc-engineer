from typing import Dict, Any, List, Optional, Protocol
import os
from dotenv import load_dotenv
from abc import ABC, abstractmethod
from types import SimpleNamespace

from .document_parser import Section, DocumentPlan, GeneratedSection


load_dotenv()


class ModelProvider(Protocol):
    """Protocol for AI model providers."""
    
    def generate_content(self, prompt: str, generation_config: Dict[str, Any]) -> Any:
        """Generate content from the model."""
        ...


class GeminiProvider:
    """Implementation of ModelProvider using Google's Gemini models."""
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-thinking-exp-01-21"):
        # Import here to avoid requiring google package for mock usage
        import google.generativeai as genai
        
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("No API key provided. Please provide an API key via constructor or GOOGLE_API_KEY environment variable.")
        
        print(f"Configuring Gemini with API key: {self.api_key[:5]}...")
        
        try:
            # First try the newer client API format
            if hasattr(genai, 'Client'):
                self.client = genai.Client(api_key=self.api_key)
                self._use_new_api = True
                print(f"Successfully initialized Gemini client with model: {model_name} (new API)")
            # Fall back to the older API format if Client isn't available
            else:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(model_name)
                self._use_new_api = False
                print(f"Successfully initialized Gemini model: {model_name} (legacy API)")
                
            self.genai = genai
        except Exception as e:
            print(f"Error initializing Gemini: {e}")
            raise
    
    def generate_content(self, prompt: str, generation_config: Dict[str, Any]) -> Any:
        """Generate content using the Gemini model."""
        try:
            # Use the appropriate API based on what was initialized
            if self._use_new_api:
                # Newer client.models.generate_content method
                response = self.client.models.generate_content(
                    model=self.model_name, 
                    contents=prompt,
                    generation_config=generation_config
                )
                # Create a wrapper to maintain compatibility with the existing code
                return SimpleNamespace(text=response.text)
            else:
                # Legacy API format
                return self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
        except Exception as e:
            print(f"Error generating content with Gemini: {e}")
            # Include a more verbose error message for debugging
            if hasattr(e, 'details'):
                print(f"Error details: {e.details}")
            raise


class MockProvider:
    """Mock implementation of ModelProvider for testing purposes."""
    
    def __init__(self):
        """Initialize the mock provider."""
        self.responses = {
            "document_plan": """
            {
                "introduction": {
                    "title": "Introduction",
                    "description": "An overview of the topic",
                    "subsections": ["Background", "Significance", "Scope"],
                    "estimated_length": 500
                },
                "main_sections": [
                    {
                        "title": "Literature Review",
                        "description": "Summary of existing research",
                        "subsections": ["Historical Context", "Current Approaches", "Gaps"],
                        "estimated_length": 1000
                    },
                    {
                        "title": "Methodology",
                        "description": "Research approach",
                        "subsections": ["Design", "Data Collection", "Analysis"],
                        "estimated_length": 1000
                    },
                    {
                        "title": "Results",
                        "description": "Key findings",
                        "subsections": ["Primary Outcomes", "Secondary Outcomes", "Correlations"],
                        "estimated_length": 1000
                    },
                    {
                        "title": "Discussion",
                        "description": "Interpretation of results",
                        "subsections": ["Significance", "Limitations", "Future Directions"],
                        "estimated_length": 1000
                    }
                ],
                "conclusion": {
                    "title": "Conclusion",
                    "description": "Summary of key points",
                    "subsections": ["Summary", "Implications", "Final Thoughts"],
                    "estimated_length": 500
                },
                "total_estimated_length": 5000
            }
            """,
            "section_content": "This is mock content for a section. It would normally be much longer and more detailed, covering the topic in depth with academic language and proper citations.",
            "consistency": "No consistency issues found.",
            "critique": """
            SECTION 0:
            The introduction provides a good overview but could be more engaging. Consider adding more context about why this topic matters.
            
            SECTION 1:
            The literature review is comprehensive but lacks critical analysis of the studies mentioned. 
            """
        }
    
    def generate_content(self, prompt: str, generation_config: Dict[str, Any]) -> Any:
        """Generate content using pre-defined mock responses."""
        # Determine which type of response to return based on prompt content
        if "Create a detailed academic document plan" in prompt:
            response = self.responses["document_plan"]
        elif "Write the content for the following section" in prompt:
            response = self.responses["section_content"]
        elif "Review the following new section for consistency" in prompt:
            response = self.responses["consistency"]
        elif "Evaluate this academic document" in prompt:
            response = self.responses["critique"]
        else:
            response = "This is a mock response."
        
        # Create a SimpleNamespace to mimic the structure of a real response
        return SimpleNamespace(text=response)


class ContentGenerator:
    """Generates content for document sections."""
    
    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider
    
    def generate_section_content(
        self,
        title: str,
        section: Section,
        previous_sections: Optional[List[GeneratedSection]] = None
    ) -> GeneratedSection:
        """Generate content for a specific section."""
        context = self._create_section_context(section, previous_sections)
        
        prompt = f"""Write the content for the following section of an academic article about "{title}":

{context}

Requirements:
1. Write approximately {section.estimated_length} words
2. Use academic language and tone
3. Structure the content according to the subsections
4. Maintain consistency with previous sections
5. Use clear topic sentences for each paragraph
6. Include appropriate transitions between ideas

Format the content in Markdown with proper headings and paragraphs.
IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
Do NOT wrap your response in ```markdown or any other code block format."""

        try:
            response = self.model_provider.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.7,
                    "max_output_tokens": 4000,
                }
            )
            
            # Clean any potential markdown code blocks from the response
            cleaned_content = self._clean_markdown_blocks(response.text)
            
            # Return the generated section
            return GeneratedSection(
                title=section.title,
                content=cleaned_content,
                subsections=section.subsections,
                level=section.level
            )
            
        except Exception as e:
            print(f"Error generating content for '{section.title}': {e}")
            # Return a minimal section in case of error
            return GeneratedSection(
                title=section.title,
                content="Error generating content for this section.",
                subsections=section.subsections,
                level=section.level
            )
    
    def check_consistency(
        self,
        content: str,
        section_title: str,
        previous_sections: List[GeneratedSection]
    ) -> str:
        """Check consistency with previous sections."""
        sections_text = "\n\n".join(
            f"Section: {section.title}\n{section.content[:300]}..."
            for section in previous_sections
        )
        
        prompt = f"""Review the following new section for consistency with previous sections:

Previous Sections:
{sections_text}

New Section: {section_title}
{content[:500]}...

Check for:
1. Logical flow and consistency with previous sections
2. Consistent terminology and definitions
3. Proper transitions between sections
4. Academic tone and style consistency

If there are no consistency issues, respond with "No consistency issues found."
If there are issues, provide a concise list of specific issues that need to be addressed.

IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
Do NOT wrap your response in ```markdown or any other code block format."""

        try:
            response = self.model_provider.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 1000,
                }
            )
            
            # Clean any potential markdown code blocks from the response
            consistency_report = self._clean_markdown_blocks(response.text)
            
            # If no issues found, return empty string
            if "No consistency issues found" in consistency_report:
                return ""
            
            return consistency_report
            
        except Exception as e:
            print(f"Error checking consistency for '{section_title}': {e}")
            return ""
    
    def evaluate_document_sections(
        self,
        title: str,
        sections: List[GeneratedSection]
    ) -> Dict[int, str]:
        """Evaluate the overall document and create specific critiques for each section."""
        # Create a preview of the entire document (with truncated sections to fit token limits)
        section_previews = []
        for i, section in enumerate(sections):
            # Truncate the content to avoid token limits
            truncated_content = section.content[:300] + "..." if len(section.content) > 300 else section.content
            section_previews.append(f"Section {i}: {section.title}\n{truncated_content}\n")
        
        document_preview = "\n".join(section_previews)
        
        prompt = f"""Evaluate this academic document about "{title}" and provide specific improvement critiques for EACH SECTION.

Document Sections:
{document_preview}

For EACH SECTION (0, 1, 2, etc.), provide:
1. Specific issues with coherence, clarity, or style
2. Content gaps or imbalances
3. Academic tone inconsistencies
4. Improvement suggestions for readability
5. Transition improvement recommendations

Format your response as follows:
SECTION 0:
[Critique and improvement suggestions for the introduction]

SECTION 1:
[Critique and improvement suggestions for main section 1]

... and so on for each section ...

Be specific and actionable with your critiques. Each section's feedback will be used to revise that section.
"""

        try:
            response = self.model_provider.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.3,
                    "max_output_tokens": 2000,
                }
            )
            
            from .document_parser import DocumentParser
            # Get the raw text from the response
            critique_text = self._clean_markdown_blocks(response.text)
            
            # Parse the critiques by section
            return DocumentParser.parse_critiques(critique_text)
            
        except Exception as e:
            print(f"Error evaluating document: {e}")
            return {}
    
    def revise_section(
        self,
        original_section: GeneratedSection,
        critique: str,
        plan_section: Section,
        previous_context: List[str]
    ) -> GeneratedSection:
        """Generate revised content for a section based on critique and original content."""
        context = "\n".join(previous_context) if previous_context else "None"
        
        prompt = f"""Revise the following section of an academic article based on critique feedback:

Section Title: {original_section.title}
Original Content: 
{original_section.content}

Critique and Improvements Needed:
{critique}

Previous Sections Context:
{context}

Section Plan:
- Description: {plan_section.description}
- Subsections: {', '.join(plan_section.subsections)}
- Estimated Length: {plan_section.estimated_length} words

Requirements:
1. Address ALL issues mentioned in the critique
2. Improve readability and academic tone
3. Ensure smooth transitions with previous sections
4. Maintain the original intent and structure
5. Stay within the approximate target length

Format the revised content in Markdown with proper paragraphs.
DO NOT include the title in your response.
IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
"""

        try:
            response = self.model_provider.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.5,
                    "max_output_tokens": 4000,
                }
            )
            
            # Clean any potential markdown code blocks from the response
            revised_content = self._clean_markdown_blocks(response.text)
            
            return GeneratedSection(
                title=original_section.title,
                content=revised_content,
                subsections=original_section.subsections
            )
            
        except Exception as e:
            print(f"Error revising section {original_section.title}: {e}")
            return original_section  # Return original if revision fails
    
    def create_document_plan(self, title: str, num_sections: int = 5, target_length_words: Optional[int] = None) -> DocumentPlan:
        """
        Create a detailed document plan with sections and subsections.
        
        Args:
            title (str): Document title
            num_sections (int): Number of sections (including intro and conclusion)
            target_length_words (Optional[int]): Target document length in words
        """
        # Set default lengths
        default_intro_length = 500
        default_main_section_length = 1000
        default_conclusion_length = 500
        
        # If target length is specified, distribute words among sections
        if target_length_words:
            # Calculate main sections (excluding intro and conclusion)
            main_sections_count = num_sections - 2
            
            # Distribute words strategically across sections
            # Introduction: 15-20% for shorter docs, 10-15% for longer docs
            intro_percent = 0.15 if target_length_words < 7500 else 0.10
            words_intro = max(300, int(target_length_words * intro_percent))
            
            # Conclusion: 15-20% for shorter docs, 10-15% for longer docs
            conclusion_percent = 0.15 if target_length_words < 7500 else 0.10
            words_conclusion = max(300, int(target_length_words * conclusion_percent))
            
            # Distribute remaining words to main sections
            remaining_words = target_length_words - words_intro - words_conclusion
            
            # If we have main sections, distribute words with emphasis on earlier sections
            if main_sections_count > 0:
                # Calculate words per main section (earlier sections get more words)
                main_section_lengths = []
                if main_sections_count == 1:
                    # Only one main section gets all remaining words
                    main_section_lengths = [remaining_words]
                else:
                    # Calculate weights for distribution
                    # First sections get more words, decreasing gradually
                    weights = []
                    for i in range(main_sections_count):
                        # Weights decrease: 1.0, 0.95, 0.9, 0.85...
                        weight = 1.0 - (i * 0.05)
                        weights.append(max(0.6, weight))
                    
                    # Normalize weights to sum to 1.0
                    total_weight = sum(weights)
                    weights = [w / total_weight for w in weights]
                    
                    # Distribute words based on weights
                    main_section_lengths = [max(400, int(remaining_words * w)) for w in weights]
                    
                    # Adjust to ensure total matches the target
                    total_allocated = sum(main_section_lengths)
                    if total_allocated != remaining_words:
                        # Distribute any difference to the sections proportionally
                        difference = remaining_words - total_allocated
                        for i in range(main_sections_count):
                            # Add a portion of the difference to each section
                            portion = int(difference * weights[i])
                            main_section_lengths[i] += portion
                            difference -= portion
                            
                            # Add any remaining difference to the last section
                            if i == main_sections_count - 1 and difference != 0:
                                main_section_lengths[i] += difference
            else:
                # No main sections (just intro and conclusion)
                main_section_lengths = []
                
                # Adjust intro and conclusion lengths to match target
                total = words_intro + words_conclusion
                if total != target_length_words:
                    # Distribute any difference proportionally
                    difference = target_length_words - total
                    intro_portion = int(difference * 0.5)
                    words_intro += intro_portion
                    words_conclusion += (difference - intro_portion)
        else:
            # Use default lengths if no target specified
            words_intro = default_intro_length
            words_conclusion = default_conclusion_length
            main_section_lengths = [default_main_section_length] * (num_sections - 2)
        
        # Calculate total length
        total_length = words_intro + words_conclusion
        if len(main_section_lengths) > 0:
            total_length += sum(main_section_lengths)
        
        # Create fallback plan with calculated lengths
        fallback_plan = {
            "introduction": {
                "title": "Introduction",
                "description": f"Introduction to {title}",
                "subsections": ["Background", "Context", "Scope"],
                "estimated_length": words_intro
            },
            "main_sections": [],
            "conclusion": {
                "title": "Conclusion",
                "description": "Summary and implications",
                "subsections": ["Summary", "Implications", "Future Directions"],
                "estimated_length": words_conclusion
            },
            "total_estimated_length": total_length
        }
        
        # Add main sections with calculated lengths
        section_names = [
            "Literature Review", "Methodology", "Results", "Discussion", 
            "Applications", "Challenges", "Case Studies", "Analysis"
        ]
        
        for i in range(num_sections - 2):
            if i < len(section_names):
                section_title = section_names[i]
            else:
                section_title = f"Section {i+1}"
            
            # Get appropriate word count
            section_length = main_section_lengths[i] if i < len(main_section_lengths) else default_main_section_length
            
            # Determine number of subsections based on length
            subsections = []
            if section_length >= 1500:
                subsections = ["Background", "Core Concepts", "Key Developments", "Analysis", "Applications", "Future Directions"]
            elif section_length >= 1000:
                subsections = ["Background", "Main Elements", "Analysis", "Applications"]
            else:
                subsections = ["Key Points", "Analysis", "Examples"]
            
            fallback_plan["main_sections"].append({
                "title": section_title,
                "description": f"Details about {section_title}",
                "subsections": subsections,
                "estimated_length": section_length
            })
        
        # Try to generate a plan with the AI, but use fallback if it fails
        try:
            # Create a detailed prompt with specific lengths for each section
            main_sections_json = []
            for i, section in enumerate(fallback_plan["main_sections"]):
                section_json = f"""
        {{
            "title": "{section['title']}",
            "description": "{section['description']}",
            "subsections": {str(section['subsections']).replace("'", '"')},
            "estimated_length": {section['estimated_length']}
        }}{"," if i < len(fallback_plan["main_sections"]) - 1 else ""}"""
                main_sections_json.append(section_json)
            
            main_sections_prompt = "".join(main_sections_json)
            
            prompt = f"""Create a document outline for a {total_length}-word article about {title}.

Please provide a JSON object with this EXACT structure:
{{
    "introduction": {{
        "title": "Introduction",
        "description": "Introduction to {title}",
        "subsections": ["Background", "Context", "Scope"],
        "estimated_length": {words_intro}
    }},
    "main_sections": [{main_sections_prompt}
    ],
    "conclusion": {{
        "title": "Conclusion",
        "description": "Summary and implications",
        "subsections": ["Summary", "Implications", "Future Directions"],
        "estimated_length": {words_conclusion}
    }},
    "total_estimated_length": {total_length}
}}

IMPORTANT: Keep the exact estimated_length values as provided. ONLY output valid JSON, nothing else."""

            response = self.model_provider.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.4,  # Lower temperature for more predictable output
                    "max_output_tokens": 1500,
                }
            )
            
            # Get the response text and clean it up
            raw_text = response.text.strip()
            
            # Simple approach - find the opening and closing braces for the full JSON object
            start = raw_text.find('{')
            end = raw_text.rfind('}')
            
            if start >= 0 and end > start:
                json_str = raw_text[start:end+1]
                
                # Use a safer JSON parsing approach
                import json
                try:
                    plan_data = json.loads(json_str)
                    
                    # Validate the plan has the required keys
                    for key in ["introduction", "main_sections", "conclusion", "total_estimated_length"]:
                        if key not in plan_data:
                            raise ValueError(f"Missing key: {key}")
                    
                    # Verify section lengths match our requirements
                    plan_data["introduction"]["estimated_length"] = words_intro
                    plan_data["conclusion"]["estimated_length"] = words_conclusion
                    
                    # Ensure main sections have correct lengths
                    for i, section in enumerate(plan_data["main_sections"]):
                        if i < len(main_section_lengths):
                            section["estimated_length"] = main_section_lengths[i]
                    
                    # Update total estimated length
                    plan_data["total_estimated_length"] = total_length
                    
                    # Create a DocumentPlan object
                    return DocumentPlan.from_dict(plan_data, title)
                    
                except json.JSONDecodeError as e:
                    print(f"JSON parsing error: {e}")
                    return DocumentPlan.from_dict(fallback_plan, title)
            else:
                print("Could not find a valid JSON object in the response")
                return DocumentPlan.from_dict(fallback_plan, title)
                
        except Exception as e:
            print(f"Error creating document plan: {e}")
            return DocumentPlan.from_dict(fallback_plan, title)
    
    def _create_section_context(
        self,
        section: Section,
        previous_sections: Optional[List[GeneratedSection]] = None
    ) -> str:
        """Create context for section generation."""
        context = f"""Section Title: {section.title}
Description: {section.description}
Subsections: {', '.join(section.subsections)}
Target Length: {section.estimated_length} words"""

        if previous_sections:
            context += "\n\nPrevious Sections:\n"
            for prev_section in previous_sections:
                # Include just a preview of previous sections to avoid token limits
                preview = prev_section.content[:200] + "..." if len(prev_section.content) > 200 else prev_section.content
                context += f"\n{prev_section.title}:\n{preview}"

        return context
    
    def _clean_markdown_blocks(self, text: str) -> str:
        """Remove any markdown code blocks from the text."""
        # Remove code blocks with language specifier
        while "```" in text:
            start = text.find("```")
            # Find the closing ```
            end = text.find("```", start + 3)
            if end > start:
                # Replace the code block with its content (if it's not empty)
                block_content = text[start+3:end].strip()
                language_end = block_content.find("\n")
                if language_end > 0:  # There's a language specifier
                    block_content = block_content[language_end:].strip()
                # Replace the code block with its content
                text = text[:start] + block_content + text[end+3:]
            else:
                break  # No closing ```, stop processing
                
        return text 