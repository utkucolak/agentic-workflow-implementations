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
print("\n--- AugmentedPromptAgent Response ---")
print(augmented_agent_response)

# KNOWLEDGE SOURCE & PERSONA IMPACT ANALYSIS:
print("\n--- Analysis ---")
print("Knowledge Source:")
print("  - The LLM's base training data provides foundational knowledge about world capitals")
print("  - No explicit domain knowledge is constrained in this agent")
print("  - The model relies on its pre-trained understanding to answer factual queries")
print()
print("Persona Impact:")
print("  - The persona system prompt forces adoption of 'college professor' identity")
print("  - This transforms response format: answers begin with 'Dear students,'")
print("  - The persona instructs the agent to explain reasoning, adding pedagogical depth")
print("  - Result: Response styled for classroom context rather than bare factual answer")
print("  - The persona overrides default communication pattern and enforces structured output")
print()
print("Key Difference from DirectPromptAgent:")
print("  - DirectPromptAgent: Uses only user prompt (no system role) → generic LLM behavior")
print("  - AugmentedPromptAgent: Injects persona as system message → tailored communication style")
