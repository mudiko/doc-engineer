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
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
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
                    # Check if response has text before accessing it
                    if hasattr(response, 'text'):
                        return SimpleNamespace(text=response.text)
                    elif hasattr(response, 'candidates') and response.candidates:
                        # Try to get text from first candidate
                        if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                            parts = response.candidates[0].content.parts
                            return SimpleNamespace(text=parts[0].text if parts else "")
                        return SimpleNamespace(text="Empty response from model")
                    else:
                        # Handle empty response
                        return SimpleNamespace(text="Empty response from model")
                else:
                    # Legacy API format
                    response = self.model.generate_content(
                        prompt,
                        generation_config=generation_config
                    )
                    
                    # Check if response has text before returning
                    if hasattr(response, 'text'):
                        return response
                    elif hasattr(response, 'candidates') and response.candidates:
                        # Try to get text from first candidate
                        if hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts'):
                            parts = response.candidates[0].content.parts
                            return SimpleNamespace(text=parts[0].text if parts else "")
                        return SimpleNamespace(text="Empty response from model")
                    else:
                        # Handle empty response
                        return SimpleNamespace(text="Empty response from model")
                    
            except Exception as e:
                error_message = str(e)
                error_details = getattr(e, 'details', '')
                
                # Check if this is an empty response error
                if "requires a single candidate" in error_message or "candidates` is empty" in error_message:
                    print(f"Received empty response from model. Retrying...")
                    retry_count += 1
                    wait_time = 10  # Wait 10 seconds before retrying for empty response
                    import time
                    time.sleep(wait_time)
                    continue
                # Check if this is a quota error (HTTP 429)
                elif "429" in error_message or "quota" in error_message.lower() or "exhausted" in error_message.lower():
                    retry_count += 1
                    wait_time = 60  # Wait for 60 seconds on quota error
                    
                    # Try to extract retry delay if available
                    import re
                    retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', str(error_details))
                    if retry_match:
                        wait_time = int(retry_match.group(1))
                    
                    if retry_count < max_retries:
                        print(f"Quota limit reached. Waiting for {wait_time} seconds before retry ({retry_count}/{max_retries})...")
                        import time
                        time.sleep(wait_time)
                        continue
                    else:
                        print(f"Maximum retries reached ({max_retries}). Giving up.")
                
                # For other errors or after max retries
                print(f"Error generating content with Gemini: {e}")
                # Include a more verbose error message for debugging
                if error_details:
                    print(f"Error details: {error_details}")
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
            """,
            "revised_section": "This is improved content for the section based on critiques. It would normally be much longer and more detailed, with improvements addressing the specific issues raised in the critique."
        }
    
    def generate_content(self, prompt: str, generation_config: Dict[str, Any]) -> Any:
        """Generate content using pre-defined mock responses."""
        # Determine which type of response to return based on prompt content
        if "Create a document outline" in prompt:
            response = self.responses["document_plan"]
        elif "Write the content for the following section" in prompt:
            response = self.responses["section_content"]
        elif "Review the following new section for consistency" in prompt:
            response = self.responses["consistency"]
        elif "Evaluate this academic document" in prompt:
            response = self.responses["critique"]
        elif "Revise the following section" in prompt:
            response = self.responses["revised_section"]
        else:
            response = "This is a mock response."
        
        # Create a SimpleNamespace to mimic the structure of a real response
        return SimpleNamespace(text=response)


class ContentGenerator:
    """Generates content for document sections."""
    
    def __init__(self, model_provider: ModelProvider):
        self.model_provider = model_provider
        # Approximate words per token for content generation
        self.words_per_token = 0.6
        # Maximum tokens per generation request (approximately 5000 words)
        self.max_output_tokens = 8192
    
    def generate_section_content(
        self,
        title: str,
        section: Section,
        previous_sections: Optional[List[GeneratedSection]] = None
    ) -> GeneratedSection:
        """Generate content for a specific section."""
        context = self._create_section_context(section, previous_sections)
        
        # Check if the section needs chunking based on estimated length
        if section.estimated_length > self._tokens_to_words(self.max_output_tokens):
            return self._generate_chunked_section_content(title, section, context, previous_sections)
        
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

        content = self._safely_generate_content(
            prompt,
            temperature=0.7,
            max_output_tokens=min(self._words_to_tokens(section.estimated_length), self.max_output_tokens),
            purpose=f"generating content for section '{section.title}'"
        )
        
        # Clean any potential markdown code blocks from the response
        cleaned_content = self._clean_markdown_blocks(content)
        
        # Return the generated section
        return GeneratedSection(
            title=section.title,
            content=cleaned_content,
            subsections=section.subsections,
            level=section.level
        )
    
    def _generate_chunked_section_content(
        self,
        title: str,
        section: Section,
        context: str,
        previous_sections: Optional[List[GeneratedSection]] = None
    ) -> GeneratedSection:
        """Generate content for a large section in chunks and combine them."""
        # Calculate number of chunks needed
        max_words_per_chunk = self._tokens_to_words(self.max_output_tokens)
        num_chunks = (section.estimated_length + max_words_per_chunk - 1) // max_words_per_chunk
        
        print(f"Chunking section '{section.title}' into {num_chunks} chunks (estimated length: {section.estimated_length} words)")
        
        # Adjust subsections if needed for chunking
        if section.subsections and len(section.subsections) >= num_chunks:
            # Use subsections to guide chunking
            chunks_plan = self._plan_chunks_by_subsections(section.subsections, num_chunks)
        else:
            # Create generic chunk descriptions
            chunks_plan = [f"Part {i+1} of {num_chunks}" for i in range(num_chunks)]
        
        all_content = []
        previous_chunk_content = ""
        
        for i, chunk_desc in enumerate(chunks_plan):
            is_first_chunk = i == 0
            is_last_chunk = i == len(chunks_plan) - 1
            
            # Calculate target word count for this chunk
            chunk_word_count = section.estimated_length // num_chunks
            if is_last_chunk:
                # Adjust last chunk to account for any remaining words
                chunk_word_count = section.estimated_length - (chunk_word_count * (num_chunks - 1))
            
            # Create prompt for this chunk
            chunk_prompt = self._create_chunk_prompt(
                title=title,
                section=section,
                chunk_desc=chunk_desc, 
                chunk_index=i,
                total_chunks=num_chunks,
                word_count=chunk_word_count,
                previous_chunk_content=previous_chunk_content,
                context=context if is_first_chunk else None
            )
            
            chunk_content = self._safely_generate_content(
                chunk_prompt,
                temperature=0.7,
                max_output_tokens=min(self._words_to_tokens(chunk_word_count * 1.2), self.max_output_tokens),
                purpose=f"generating chunk {i+1}/{num_chunks} for section '{section.title}'"
            )
            
            chunk_content = self._clean_markdown_blocks(chunk_content)
            all_content.append(chunk_content)
            previous_chunk_content = chunk_content[-1000:] if len(chunk_content) > 1000 else chunk_content
        
        # Combine all chunks into a single content
        combined_content = "\n\n".join(all_content)
        
        return GeneratedSection(
            title=section.title,
            content=combined_content,
            subsections=section.subsections,
            level=section.level
        )
    
    def _create_chunk_prompt(
        self, 
        title: str,
        section: Section,
        chunk_desc: str, 
        chunk_index: int,
        total_chunks: int,
        word_count: int,
        previous_chunk_content: str = "",
        context: Optional[str] = None
    ) -> str:
        """Create a prompt for generating a specific chunk of content."""
        
        if chunk_index == 0:
            # First chunk
            prompt = f"""Write the beginning part of the section "{section.title}" for an academic article about "{title}":

{context}

This is part 1 of {total_chunks} for this section.

Requirements:
1. Write approximately {word_count} words
2. Use academic language and tone
3. Begin the section appropriately with an introduction
4. Focus on: {chunk_desc}
5. Remember this is only the first part - don't try to conclude the section

Format the content in Markdown with proper headings and paragraphs.
IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
Do NOT wrap your response in ```markdown or any other code block format."""
        
        elif chunk_index == total_chunks - 1:
            # Last chunk
            prompt = f"""Write the final part of the section "{section.title}" for an academic article about "{title}":

This is part {chunk_index+1} of {total_chunks} for this section.

Previous part ended with:
{previous_chunk_content}

Requirements:
1. Write approximately {word_count} words
2. Use academic language and tone
3. Focus on: {chunk_desc}
4. Provide a proper conclusion to the entire section
5. Ensure smooth continuation from the previous part

Format the content in Markdown with proper headings and paragraphs.
IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
Do NOT wrap your response in ```markdown or any other code block format."""
        
        else:
            # Middle chunk
            prompt = f"""Write the middle part of the section "{section.title}" for an academic article about "{title}":

This is part {chunk_index+1} of {total_chunks} for this section.

Previous part ended with:
{previous_chunk_content}

Requirements:
1. Write approximately {word_count} words
2. Use academic language and tone
3. Focus on: {chunk_desc}
4. Ensure smooth continuation from the previous part
5. Remember this is not the conclusion - continue the discussion

Format the content in Markdown with proper headings and paragraphs.
IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
Do NOT wrap your response in ```markdown or any other code block format."""
        
        return prompt
    
    def _plan_chunks_by_subsections(self, subsections: List[str], num_chunks: int) -> List[str]:
        """Plan how to distribute subsections across chunks."""
        if len(subsections) <= num_chunks:
            return subsections
        
        # Group subsections into chunks
        chunks = []
        subsections_per_chunk = len(subsections) / num_chunks
        
        for i in range(num_chunks):
            start_idx = int(i * subsections_per_chunk)
            end_idx = int((i + 1) * subsections_per_chunk)
            if i == num_chunks - 1:
                end_idx = len(subsections)  # Ensure we include all remaining subsections
            
            chunk_subsections = subsections[start_idx:end_idx]
            if len(chunk_subsections) == 1:
                chunks.append(chunk_subsections[0])
            else:
                chunks.append(", ".join(chunk_subsections))
        
        return chunks
    
    def _words_to_tokens(self, word_count: int) -> int:
        """Convert word count to approximate token count."""
        return int(word_count / self.words_per_token)
    
    def _tokens_to_words(self, token_count: int) -> int:
        """Convert token count to approximate word count."""
        return int(token_count * self.words_per_token)
    
    def check_consistency(
        self,
        content: str,
        section_title: str,
        previous_sections: List[GeneratedSection]
    ) -> str:
        """Check consistency with previous sections."""
        # Estimate the total size of content to check
        total_content_size = len(content) + sum(len(section.content) for section in previous_sections)
        
        # If content is too large, use a summary approach
        if total_content_size > 50000:  # Approximately 10k words
            return self._check_consistency_summarized(content, section_title, previous_sections)
        
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
If there are issues, describe them clearly and concisely, focusing on the most important ones first."""

        return self._safely_generate_content(
            prompt,
            temperature=0.3,
            max_output_tokens=min(2000, self.max_output_tokens),
            purpose=f"checking consistency for section '{section_title}'"
        )
    
    def _check_consistency_summarized(
        self,
        content: str,
        section_title: str,
        previous_sections: List[GeneratedSection]
    ) -> str:
        """Check consistency with a summarization approach for large documents."""
        # First, generate summaries of previous sections
        section_summaries = []
        
        # Group sections if there are many
        if len(previous_sections) > 5:
            # Process sections in smaller groups
            for i in range(0, len(previous_sections), 3):
                group = previous_sections[i:i+3]
                group_text = "\n\n".join(
                    f"Section: {section.title}\n{section.content[:200]}..."
                    for section in group
                )
                
                summary_prompt = f"""Summarize the key points, terminology, and style of these document sections:

{group_text}

Create a concise summary (150-200 words) that captures:
1. Main topics and themes
2. Key terminology and definitions
3. Academic tone and approach
4. The logical flow between these sections"""

                summary_text = self._safely_generate_content(
                    summary_prompt,
                    temperature=0.3,
                    max_output_tokens=1000,
                    purpose=f"summary for sections {i+1}-{i+len(group)}"
                )
                
                section_summaries.append(f"Sections {i+1}-{i+len(group)}:\n{summary_text}")
        else:
            # Process each section individually if there aren't too many
            for i, section in enumerate(previous_sections):
                summary_prompt = f"""Summarize the key points, terminology, and style of this document section:

Section: {section.title}
{section.content[:300]}...

Create a concise summary (100 words) that captures:
1. Main topics and themes
2. Key terminology and definitions
3. Academic tone and approach"""

                summary_text = self._safely_generate_content(
                    summary_prompt,
                    temperature=0.3,
                    max_output_tokens=500,
                    purpose=f"summary for section {section.title}"
                )
                
                section_summaries.append(f"Section: {section.title}\n{summary_text}")
        
        # Now check consistency with the summaries
        summaries_text = "\n\n".join(section_summaries)
        
        consistency_prompt = f"""Review the following new section for consistency with previous sections (summarized below):

Previous Sections (Summarized):
{summaries_text}

New Section: {section_title}
{content[:500]}...

Check for:
1. Logical flow and consistency with previous sections
2. Consistent terminology and definitions
3. Proper transitions between sections
4. Academic tone and style consistency

If there are no consistency issues, respond with "No consistency issues found."
If there are issues, describe them clearly and concisely, focusing on the most important ones first."""

        consistency_text = self._safely_generate_content(
            consistency_prompt,
            temperature=0.3,
            max_output_tokens=min(2000, self.max_output_tokens),
            purpose="consistency check with summaries"
        )
        
        return consistency_text
    
    def _safely_generate_content(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_output_tokens: int = 1000,
        purpose: str = "content"
    ) -> str:
        """
        Generate content with error handling, returning a default message if generation fails.
        
        Args:
            prompt (str): The prompt to send to the model
            temperature (float): Generation temperature
            max_output_tokens (int): Maximum tokens to generate
            purpose (str): Description of what's being generated (for error messages)
            
        Returns:
            str: Generated content or error message
        """
        try:
            response = self.model_provider.generate_content(
                prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
            )
            
            return response.text
            
        except Exception as e:
            error_message = f"Error generating {purpose}: {str(e)}"
            print(error_message)
            return f"[{error_message}]"
    
    def evaluate_document_sections(
        self,
        title: str,
        sections: List[GeneratedSection]
    ) -> Dict[int, str]:
        """Evaluate and critique document sections."""
        # If document is too large, evaluate in chunks
        total_content_length = sum(len(section.content) for section in sections)
        approximate_word_count = total_content_length / 5  # Approximate 5 chars per word
        
        if approximate_word_count > self._tokens_to_words(self.max_output_tokens * 0.8):
            return self._evaluate_document_sections_chunked(title, sections)
        
        sections_text = "\n\n".join(
            f"SECTION {i}: {section.title}\n{section.content}"
            for i, section in enumerate(sections)
        )
        
        prompt = f"""Evaluate this academic document about "{title}" and provide a critique for each section:

{sections_text}

For EACH numbered section, provide a critique that addresses:
1. Content quality and depth
2. Organization and structure
3. Academic tone and language
4. Clarity and precision
5. Specific suggestions for improvement

Format your response with section numbers:
SECTION 0:
[Your critique of section 0]

SECTION 1:
[Your critique of section 1]

And so on for each section.

Focus on constructive feedback that will help improve the quality of the document."""

        response_text = self._safely_generate_content(
            prompt,
            temperature=0.3,
            max_output_tokens=min(self._words_to_tokens(approximate_word_count * 0.5), self.max_output_tokens),
            purpose="document evaluation"
        )
        
        # Parse the critique response into a dictionary
        critiques = {}
        current_section = None
        current_critique = []
        
        for line in response_text.split('\n'):
            if line.startswith('SECTION '):
                # Save previous section if exists
                if current_section is not None and current_critique:
                    critiques[current_section] = '\n'.join(current_critique)
                    current_critique = []
                
                # Extract section number
                try:
                    section_text = line.split(':')[0].strip()
                    current_section = int(section_text.replace('SECTION ', ''))
                except (ValueError, IndexError):
                    print(f"Warning: Couldn't parse section number from line: {line}")
                    current_section = None
            elif current_section is not None:
                current_critique.append(line)
        
        # Add the last section
        if current_section is not None and current_critique:
            critiques[current_section] = '\n'.join(current_critique)
        
        return critiques

    def _evaluate_document_sections_chunked(
        self,
        title: str,
        sections: List[GeneratedSection]
    ) -> Dict[int, str]:
        """Evaluate document sections in chunks when the document is too large."""
        critiques = {}
        
        # Determine a reasonable chunk size (max 3-4 sections per chunk)
        max_sections_per_chunk = 3
        section_chunks = []
        
        for i in range(0, len(sections), max_sections_per_chunk):
            section_chunks.append(sections[i:i + max_sections_per_chunk])
        
        print(f"Evaluating document in {len(section_chunks)} chunks")
        
        for chunk_index, chunk_sections in enumerate(section_chunks):
            chunk_critiques = self._evaluate_section_chunk(title, chunk_sections, chunk_index, len(section_chunks))
            
            # Calculate the correct section indices
            base_index = chunk_index * max_sections_per_chunk
            for relative_idx, critique in chunk_critiques.items():
                absolute_idx = base_index + relative_idx
                if absolute_idx < len(sections):
                    critiques[absolute_idx] = critique
        
        return critiques

    def _evaluate_section_chunk(
        self,
        title: str,
        chunk_sections: List[GeneratedSection],
        chunk_index: int,
        total_chunks: int
    ) -> Dict[int, str]:
        """Evaluate a chunk of sections."""
        sections_text = "\n\n".join(
            f"SECTION {i}: {section.title}\n{section.content}"
            for i, section in enumerate(chunk_sections)
        )
        
        prompt = f"""Evaluate this chunk ({chunk_index+1} of {total_chunks}) of an academic document about "{title}" and provide a critique for each section:

{sections_text}

For EACH numbered section, provide a critique that addresses:
1. Content quality and depth
2. Organization and structure
3. Academic tone and language
4. Clarity and precision
5. Specific suggestions for improvement

Format your response with section numbers:
SECTION 0:
[Your critique of section 0]

SECTION 1:
[Your critique of section 1]

And so on for each section.

Focus on constructive feedback that will help improve the quality of the document.
IMPORTANT: Use the SECTION numbers as given in this prompt (starting from 0 for the first section in this chunk)."""

        approximate_word_count = sum(len(section.content) for section in chunk_sections) / 5
        response_text = self._safely_generate_content(
            prompt,
            temperature=0.3,
            max_output_tokens=min(self._words_to_tokens(approximate_word_count * 0.5), self.max_output_tokens),
            purpose=f"evaluating document chunk {chunk_index+1}/{total_chunks}"
        )
        
        # Parse the critique response into a dictionary
        critiques = {}
        current_section = None
        current_critique = []
        
        for line in response_text.split('\n'):
            if line.startswith('SECTION '):
                # Save previous section if exists
                if current_section is not None and current_critique:
                    critiques[current_section] = '\n'.join(current_critique)
                    current_critique = []
                
                # Extract section number
                try:
                    section_text = line.split(':')[0].strip()
                    current_section = int(section_text.replace('SECTION ', ''))
                except (ValueError, IndexError):
                    print(f"Warning: Couldn't parse section number from line: {line}")
                    current_section = None
            elif current_section is not None:
                current_critique.append(line)
        
        # Add the last section
        if current_section is not None and current_critique:
            critiques[current_section] = '\n'.join(current_critique)
        
        return critiques

    def revise_section(
        self,
        original_section: GeneratedSection,
        critique: str,
        plan_section: Section,
        previous_context: List[str]
    ) -> GeneratedSection:
        """Revise a section based on critique."""
        # Estimate word count of the original section
        approximate_word_count = len(original_section.content) / 5
        
        # If section is too large, use chunked revision
        if approximate_word_count > self._tokens_to_words(self.max_output_tokens * 0.7):
            return self._revise_section_chunked(original_section, critique, plan_section, previous_context)
            
        context = "\n\n".join(previous_context) if previous_context else ""
        
        prompt = f"""Revise the following section based on the critique provided.

Original Section: {original_section.title}
{original_section.content}

Critique:
{critique}

Requirements:
1. Address all issues mentioned in the critique
2. Maintain or improve the section length (approximately {approximate_word_count:.0f} words)
3. Ensure academic tone and language
4. Improve clarity and precision
5. Maintain the overall structure of subsections

Previous context (for consistency):
{context}

Provide the fully revised section content in Markdown format.
IMPORTANT: Do NOT include any code blocks or fenced code sections (```).
Do NOT wrap your response in ```markdown or any other code block format.
DO NOT include "Revised Section: {original_section.title}" or any similar heading - start directly with the content."""

        revised_content = self._safely_generate_content(
            prompt,
            temperature=0.7,
            max_output_tokens=min(self._words_to_tokens(approximate_word_count * 1.2), self.max_output_tokens),
            purpose=f"revising section '{original_section.title}'"
        )
        
        # Clean any potential markdown code blocks from the response
        revised_content = self._clean_markdown_blocks(revised_content)
        
        # Return the revised section
        return GeneratedSection(
            title=original_section.title,
            content=revised_content,
            subsections=original_section.subsections,
            level=original_section.level
        )
            
    def _revise_section_chunked(
        self,
        original_section: GeneratedSection,
        critique: str,
        plan_section: Section,
        previous_context: List[str]
    ) -> GeneratedSection:
        """Revise a large section in chunks based on critique."""
        # Estimate word count and calculate chunks
        approximate_word_count = len(original_section.content) / 5
        max_words_per_chunk = self._tokens_to_words(self.max_output_tokens * 0.7)
        num_chunks = (int(approximate_word_count) + max_words_per_chunk - 1) // max_words_per_chunk
        
        print(f"Revising section '{original_section.title}' in {num_chunks} chunks")
        
        # Split the content roughly into equal chunks - this is an approximation!
        content = original_section.content
        chunk_size = len(content) // num_chunks
        content_chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = (i + 1) * chunk_size if i < num_chunks - 1 else len(content)
            
            # Try to find sentence boundaries for better splitting
            if i > 0 and start > 0:
                # Look for sentence-ending punctuation followed by space
                for j in range(start - 1, max(start - 200, 0), -1):
                    if content[j] in ['.', '!', '?'] and j+1 < len(content) and content[j+1].isspace():
                        start = j + 2  # Start after the punctuation and space
                        break
            
            content_chunks.append(content[start:end])
        
        revised_chunks = []
        context_text = "\n\n".join(previous_context) if previous_context else ""
        
        for i, chunk in enumerate(content_chunks):
            is_first_chunk = i == 0
            is_last_chunk = i == len(content_chunks) - 1
            
            # Prepare instructions based on chunk position
            start_instruction = "Start the section appropriately" if is_first_chunk else "Ensure smooth continuation from previous part"
            end_instruction = "Provide a proper conclusion to the section" if is_last_chunk else "Don't try to conclude the section yet"
            context_instruction = f"Previous context (for consistency):\n{context_text}" if is_first_chunk and context_text else ""
            
            # Build the prompt
            chunk_prompt = f"""Revise the following part ({i+1}/{num_chunks}) of section "{original_section.title}" based on the critique provided.

Original Section Part:
{chunk}

Critique:
{critique}

Requirements:
1. Address relevant issues mentioned in the critique
2. Maintain or improve the part length
3. Ensure academic tone and language
4. Improve clarity and precision
5. {start_instruction}
6. {end_instruction}

{context_instruction}

Provide the revised content for this part only.
IMPORTANT: Do NOT include any code blocks or fenced code sections.
Do NOT wrap your response in any headings or formats - start directly with the content."""

            revised_chunk = self._safely_generate_content(
                chunk_prompt,
                temperature=0.7,
                max_output_tokens=self.max_output_tokens,
                purpose=f"revising chunk {i+1}/{num_chunks} for '{original_section.title}'"
            )
            
            revised_chunk = self._clean_markdown_blocks(revised_chunk)
            revised_chunks.append(revised_chunk)
        
        # Combine all revised chunks
        revised_content = "\n\n".join(revised_chunks)
        
        return GeneratedSection(
            title=original_section.title,
            content=revised_content,
            subsections=original_section.subsections,
            level=original_section.level
        )
    
    def create_document_plan(self, title: str, num_sections: int = 5, target_length_words: Optional[int] = None) -> DocumentPlan:
        """Generate a document plan with sections and subsections."""
        # Use the default target length if none provided
        target_length = target_length_words or 4000
        
        prompt = f"""Create a document outline for an academic article about "{title}".

Requirements:
1. Create an organized plan with {num_sections} main sections (plus introduction and conclusion)
2. For each section, provide:
   - A descriptive title
   - A brief description of what that section will cover
   - 2-4 subsections as bullet points
   - Estimated length in words
3. Total document length should be approximately {target_length} words
4. Structure should follow academic norms with:
   - Introduction section
   - {num_sections} main body sections
   - Conclusion section
5. Main body sections should be titled appropriately for an academic article on this topic
6. Subsections should provide logical organization for each section

Provide your response as a JSON object with this structure:
{{
  "introduction": {{
    "title": "Introduction",
    "description": "Description of introduction content",
    "subsections": ["Subsection 1", "Subsection 2", ...],
    "estimated_length": wordCount
  }},
  "main_sections": [
    {{
      "title": "Section Title",
      "description": "Description of section content",
      "subsections": ["Subsection 1", "Subsection 2", ...],
      "estimated_length": wordCount
    }},
    ...
  ],
  "conclusion": {{
    "title": "Conclusion",
    "description": "Description of conclusion content",
    "subsections": ["Subsection 1", "Subsection 2", ...],
    "estimated_length": wordCount
  }},
  "total_estimated_length": totalWordCount
}}"""

        try:
            json_text = self._safely_generate_content(
                prompt,
                temperature=0.7,
                max_output_tokens=min(4000, self.max_output_tokens),
                purpose="creating document plan"
            )
            
            import json
            
            # Clean and parse the JSON
            json_text = self._extract_json(json_text)
            plan_data = json.loads(json_text)
            
            # Create the document plan with appropriate parsing
            from .document_parser import DocumentPlan, Section
            
            # Parse introduction
            intro_data = plan_data["introduction"]
            introduction = Section(
                title=intro_data["title"],
                description=intro_data["description"],
                subsections=intro_data["subsections"],
                estimated_length=intro_data["estimated_length"],
                level=1
            )
            
            # Parse main sections
            main_sections = []
            for i, section_data in enumerate(plan_data["main_sections"]):
                main_sections.append(
                    Section(
                        title=section_data["title"],
                        description=section_data["description"],
                        subsections=section_data["subsections"],
                        estimated_length=section_data["estimated_length"],
                        level=1
                    )
                )
            
            # Parse conclusion
            conclusion_data = plan_data["conclusion"]
            conclusion = Section(
                title=conclusion_data["title"],
                description=conclusion_data["description"],
                subsections=conclusion_data["subsections"],
                estimated_length=conclusion_data["estimated_length"],
                level=1
            )
            
            # Get the total estimated length
            total_length = plan_data.get("total_estimated_length", sum(
                s.estimated_length for s in [introduction] + main_sections + [conclusion]
            ))
            
            # Create and return the document plan
            return DocumentPlan(
                topic=title,
                introduction=introduction,
                main_sections=main_sections,
                conclusion=conclusion,
                total_estimated_length=total_length
            )
            
        except Exception as e:
            print(f"Error creating document plan: {e}")
            # Create a default document plan as fallback
            return self._create_default_document_plan(title, num_sections, target_length)
    
    def _create_default_document_plan(self, title: str, num_sections: int = 5, target_length: int = 4000) -> DocumentPlan:
        """Create a default document plan when the normal creation fails."""
        from .document_parser import DocumentPlan, Section
        
        # Calculate section lengths
        intro_length = int(target_length * 0.1)
        conclusion_length = int(target_length * 0.1)
        main_section_length = int((target_length - intro_length - conclusion_length) / num_sections)
        
        # Create introduction
        introduction = Section(
            title="Introduction",
            description="An introduction to the topic.",
            subsections=["Background", "Overview", "Scope"],
            estimated_length=intro_length,
            level=1
        )
        
        # Create main sections
        main_sections = []
        for i in range(num_sections):
            main_sections.append(
                Section(
                    title=f"Section {i+1}",
                    description=f"Content for section {i+1}.",
                    subsections=[f"Subsection {i+1}.{j+1}" for j in range(3)],
                    estimated_length=main_section_length,
                    level=1
                )
            )
        
        # Create conclusion
        conclusion = Section(
            title="Conclusion",
            description="A conclusion to the topic.",
            subsections=["Summary", "Implications", "Future Directions"],
            estimated_length=conclusion_length,
            level=1
        )
        
        # Create and return the document plan
        return DocumentPlan(
            topic=title,
            introduction=introduction,
            main_sections=main_sections,
            conclusion=conclusion,
            total_estimated_length=target_length
        )
        
    def _extract_json(self, text: str) -> str:
        """Extract JSON from a text that might contain other content."""
        import re
        
        # First, try to find JSON between code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_match:
            return json_match.group(1)
        
        # If no code blocks, look for text that looks like JSON (starts with { and ends with })
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            return json_match.group(1)
        
        # If still no match, return the original text
        return text
    
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