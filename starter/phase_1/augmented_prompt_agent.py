import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent
# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"

# PERSONA IMPACT:
# The persona parameter acts as a system-level constraint that shapes the agent's communication style,
# tone, expertise level, and behavior. Unlike a simple user query, the persona is injected as a 
# system message that persists across the entire interaction, fundamentally influencing how the 
# agent processes and responds to the user's prompt.
persona = "You are a college professor; your answers ALWAYS start with: 'Dear students,' and explain your reasoning."

augmented_agent = AugmentedPromptAgent(openai_api_key, persona)
augmented_agent_response = augmented_agent.respond(prompt)
# Print the agent's response
print(augmented_agent_response)

# KNOWLEDGE SOURCE & PERSONA IMPACT ANALYSIS:
# ============================================
# Knowledge Source:
#   - The LLM's base training data provides the foundational knowledge about world capitals
#   - No explicit domain knowledge is constrained or augmented in this agent (unlike KnowledgeAugmentedPromptAgent)
#   - The model relies on its pre-trained understanding to answer the factual query
#
# Persona Impact:
#   - The persona system prompt forces the agent to adopt a "college professor" identity
#   - This transforms the response format: answers begin with "Dear students," (mandatory greeting)
#   - The persona also instructs the agent to explain reasoning, adding pedagogical depth
#   - Result: A response styled for education/classroom context rather than a bare factual answer
#   - The persona overrides the agent's default communication pattern and enforces structured output
#
# Key Difference from DirectPromptAgent:
#   - DirectPromptAgent: Uses only user prompt (no system role) → generic LLM behavior
#   - AugmentedPromptAgent: Injects persona as system message → tailored communication style
