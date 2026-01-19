import os
from dotenv import load_dotenv
from workflow_agents.base_agents import AugmentedPromptAgent
# Load environment variables from .env file
load_dotenv()

# Retrieve OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers ALWAYS start with: 'Dear students,' and explain your reasoning."

augmented_agent = AugmentedPromptAgent(openai_api_key, persona)
augmented_agent_response = augmented_agent.respond(prompt)
# Print the agent's response
print(augmented_agent_response)

# The agent used the knowledge that the capital of France is Paris.
# The system prompt influenced the agent to respond in the style of a college professor, starting with "Dear students," and providing a detailed explanation.
