# agentic_workflow.py

import sys
import os
from pathlib import Path

# Add the starter directory to Python path for imports
starter_dir = Path(__file__).parent.parent
sys.path.insert(0, str(starter_dir))

from phase_1.workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

spec_path = Path(__file__).parent / "Product-Spec-Email-Router.txt"
product_spec = open(spec_path, "r").read()
# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
action_planning_agent = ActionPlanningAgent(openai_api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    "\n".join(product_spec.split('\n'))
)
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent
persona_evaluation_agent = "You are an evaluation manager. You need to evaluate the given prompt to determine if it fits the required criteria"
product_evaluation_criteria = "The answer should be user stories that follow this exact structure: " \
                      "As a [type of user], I want [an action or feature] so that [benefit/value]."
# Instantiate product_manager_evaluation_agent using 'persona_evaluation_agent', 'evaluation_criteria', and 'product_manager_knowledge_agent'
product_manager_evaluation_agent = EvaluationAgent(
    openai_api_key,
    persona_evaluation_agent,
    product_evaluation_criteria,
    product_manager_knowledge_agent,
    10
)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
# (This is a necessary step before TODO 8. Students should add the instantiation code here.)
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_program_manager, knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."

program_evaluation_criteria = "The answer should be product features that follow the following structure: " \
                      "Feature Name: A clear, concise title that identifies the capability\n" \
                      "Description: A brief explanation of what the feature does and its purpose\n" \
                      "Key Functionality: The specific capabilities or actions the feature provides\n" \
                      "User Benefit: How this feature creates value for the user"
program_manager_evaluation_agent = EvaluationAgent(openai_api_key, persona_program_manager_eval, program_evaluation_criteria, program_manager_knowledge_agent, 10)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
# (This is a necessary step before TODO 9. Students should add the instantiation code here.)
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(openai_api_key, persona_dev_engineer, knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
dev_evaluation_criteria = "The answer should be tasks following this exact structure: " \
                      "Task ID: A unique identifier for tracking purposes\n" \
                      "Task Title: Brief description of the specific development work\n" \
                      "Related User Story: Reference to the parent user story\n" \
                      "Description: Detailed explanation of the technical work required\n" \
                      "Acceptance Criteria: Specific requirements that must be met for completion\n" \
                      "Estimated Effort: Time or complexity estimation\n" \
                      "Dependencies: Any tasks that must be completed first"
development_engineer_evaluation_agent = EvaluationAgent(openai_api_key, persona_dev_engineer_eval, dev_evaluation_criteria, development_engineer_knowledge_agent, 10)

# Routing Agent
def product_manager_support_function(query):
    # We pass the query directly to evaluate, which handles the worker_agent internally
    result = product_manager_evaluation_agent.evaluate(query)
    return result["final_response"]

def program_manager_support_function(query):
    result = program_manager_evaluation_agent.evaluate(query)
    return result["final_response"]

def development_engineer_support_function(query):
    result = development_engineer_evaluation_agent.evaluate(query)
    return result["final_response"]

# --- TODO 10: Instantiate Routing Agent ---

# Define the routes (agents list)
# Note: "func" now points to the SUPPORT functions defined above, 
# ensuring the evaluation loop is triggered, not just a raw response.
agents_list = [
    {
        "name": "product manager agent",
        "description": "Define user stories, market requirements, and product vision.",
        "func": product_manager_support_function 
    },
    {
        "name": "program manager agent",
        "description": "Define features, timelines, project management, and prioritization.",
        "func": program_manager_support_function
    },
    {
        "name": "development engineer agent",
        "description": "Define development tasks, technical architecture, coding, and implementation details.",
        "func": development_engineer_support_function
    }
]
routing_agent = RoutingAgent(openai_api_key, agents_list)
# Each support function should:
#   1. Take the input query (e.g., a step from the action plan).
#   2. Get a response from the respective Knowledge Augmented Prompt Agent.
#   3. Have the response evaluated by the corresponding Evaluation Agent.
#   4. Return the final validated response.

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
completed_steps = []

for step in workflow_steps:
    print(f"Executing step: {step}")
    result = routing_agent.route_prompts(step)
    completed_steps.append(result)
    print(f"Result: {result}")

if completed_steps:
    print(f"Final Output: {completed_steps[-1]}")