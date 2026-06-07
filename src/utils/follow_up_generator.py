"""
Generic Follow-Up Question Generator Utility

This utility automatically generates contextual follow-up questions based on 
the user's query, agent's response, and STRICT agent capabilities.
"""

import logging
from typing import Dict, List
from litellm import acompletion

logger = logging.getLogger(__name__)

async def generate_follow_up_questions(
    user_query: str,
    response: str,
    agent_capabilities: List[str],
    context: Dict,
    llm_config: dict,
    num_questions: int = 3
) -> str:
    """
    Generate contextual follow-up questions based on STRICT agent capabilities.
    
    Args:
        user_query: The user's original query/request
        response: The agent's response text
        agent_capabilities: List of EXACT capabilities the agent has
        context: Dictionary with context about the operation performed
        llm_config: LLM configuration for acompletion
        num_questions: Number of questions to generate (default: 3)
    
    Returns:
        Comma-separated string of follow-up questions
    """
    try:
        # Build context string from dictionary
        context_str = "\n".join([f"- {k}: {v}" for k, v in context.items() if v])
        
        # Build capabilities string
        capabilities_str = "\n".join([f"- {cap}" for cap in agent_capabilities])
        
        prompt = f"""You are generating follow-up questions for an agent with STRICT, LIMITED capabilities.

AGENT'S EXACT CAPABILITIES (DO NOT SUGGEST ANYTHING OUTSIDE THIS LIST):
{capabilities_str}

USER'S QUERY:
{user_query}

OPERATION CONTEXT:
{context_str}

AGENT RESPONSE:
{response[:1000]}

Generate exactly {num_questions} follow-up commands that:
1. ONLY suggest actions the agent CAN do (from the capabilities list above)
2. Are DIFFERENT from what was just done
3. Are specific and actionable
4. Are RELEVANT to the user's original query - suggest variations of what they just asked for
5. Explore logical next steps or related operations based on what they requested
6. Use DIRECT COMMAND format - start with action verbs like "Generate", "Create", "Convert", "Analyze" (not "Can you")

CRITICAL RULES - FOLLOW STRICTLY:
- DO NOT suggest capabilities NOT in the list above
- DO NOT ask the agent to review, analyze, or explain its output unless that's in capabilities
- DO NOT ask the agent to do anything except what's in the capabilities list
- DO NOT suggest the exact same operation that was just done
- Questions must be REALISTIC and EXECUTABLE by this agent
- Questions must be RELEVANT to what the user just asked for
- Use COMMAND format starting with action verbs, not questions

EXAMPLES OF GOOD FOLLOW-UPS (adapt to user's actual query and agent capabilities):
- "Generate [variation] with a more formal tone"
- "Create [related item] for a different scenario"
- "Convert [item] to [another format]"
- "Analyze [related data] with different parameters"

EXAMPLES OF BAD FOLLOW-UPS:
- "Can you review this output?" (question format + NOT in capabilities)
- "Can you explain how you did this?" (question format + NOT in capabilities)
- Suggesting unrelated operations (e.g., resignation email when user asked for leave request)
- Repeating the exact same operation

Return ONLY {num_questions} commands separated by commas, no numbering, no extra text.
Each command must NOT contain commas; use commas only as separators between the three commands.
Format: command1, command2, command3"""

        messages = [{"role": "user", "content": prompt}]
        
        response_obj = await acompletion(
            messages=messages,
            **llm_config
        )
        
        questions = response_obj.choices[0].message.content.strip()
        
        logger.info(f"Generated follow-up questions: {questions}")
        return questions
        
    except Exception as e:
        logger.error(f"Error generating follow-up questions: {e}")
        return ""
