from utils import (
    observation_state_to_node,
    local_history,
    append_history,
    select_expansion,
    langgraph_node_to_message,
)

from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode

from data_model import DARSState, DARSNode
from config import DARS_CONFIG

##### NODES #######
def setup(state: DARSState):
  env_state = get_env_state(state)
  observation = "" # fixme-----------------------------
  current_node = observation_state_to_node(env_state, observation)
  current_node._depth = 0

  return {
    "current_node": current_node,
    "node_stack": [current_node],
  }


# this is a conditional edge
def should_expand(state: DARSState):
  current_node = state.current_node
  if current_node is None or current_node.is_terminal or current_node._depth >= DARS_CONFIG.MAX_DEPTH:
      # we should start exploring a new branch
        if state.node_stack:
            current_node = state.node_stack.pop(0)
            assert current_node.children > 0
            return Send("expand", {"current_node" : current_node})
        else:
            return "__end__"

  assert current_node.children == 0
  return Send("forward")


def forward(state: DARSState):
  current_node = state.current_node
  assert current_node.role == "user"
  conversation = langgraph_node_to_message(current_node)


  # query the solver
  message_with_tool_call = DARS_CONFIG.model({
      "conversation" : conversation
  })
  tool_node = ToolNode()#[ tool: List 
  tool_output = tool_node.invoke({"messages": [message_with_tool_call]})["messages"]
  observation = tool_output.content

  output = message_with_tool_call
  assistant_node = append_history(
      {
           "role": "assistant",
            "content": f"THOUGHT: {output.thought}\nACTION: {output.action}",
            "thought": output.thought,
            "action": output.action,
            "agent": self.name,
      },
      parent_node=current_node,
  )

  state = None #env.communicate(self.state_command) if self.state_command else None fixme-------------------
  if len(current_node.children) == 1:
      if_expand = assistant_node.should_expand(DARS_CONFIG)
      if if_expand:
          assistant_node._action_expansion_limit[assistant_node.action.split()[0]] -= 1
          for _ in range(DARS_CONFIG.num_expansions-1):
              state.node_stack.append(assistant_node.parent)
          state.node_stack.sort(key=lambda x: x._depth)

  current_node = observation_state_to_node(
      last_node=assistant_node,
      state=env_state,
      observation=observation,
      search_term=search_term,
      codegraph_context=codegraoh_context,
  )
  assert current_node.role == "user"
  return {
    "current_node": current_node,
    "node_stack": state.node_stack,
  }


def expand(state: DARSState):
  # print("--- Entering expand node ---")
  current_node = state["current_node"] # This is expected to be a 'user' node
  assert current_node.role == "user"

  # Get the assistant node whose action led to the current user node
  node_to_expand_from = current_node.parent
  assert node_to_expand_from is not None, "Cannot expand from root node or node without parent"
  assert node_to_expand_from.role == "assistant", "Parent node must be an assistant node for expansion"

  # Get configuration for expansion
  agent_name = state["self_name"]

  expansion_context = get_expansion_prompt(current_node)

  sampled_actions_info = [] # Store (thought, action, output) tuples

  # Sample candidate actions from the LLM
  # print(f"Sampling {dars_config.num_expansion_sampling} expansion candidates...")
  num_expansion_sampling = 5 # hard-coded!!!!!!!!!!!!!
  for i in range(DARS_CONFIG.num_expansion_sampling):
      # print(f"Sampling candidate {i+1}...")
      # Query the LLM with the history and expansion context
      # local_history needs access to agent name and history processor config
      query_history = local_history(current_node, expansion_context, config=state["config"], agent_name=agent_name) # Pass necessary args

      try:
          llm_output = react_agent.invoke(query_history)
          parsed_thought = llm_output.thought 
          parsed_action = llm_output.action   
          raw_output_content = llm_output.content

          sampled_actions_info.append((parsed_thought, parsed_action, raw_output_content))
          print(f"Sampled action {i+1}: {parsed_action}")

      except Exception as e:
          print(f"Error during LLM sampling for expansion candidate {i+1}: {e}")
          # Handle potential LLM errors during sampling (e.g., format errors)
          # Depending on your design, you might retry, skip this candidate, or terminate the branch.
          # For now, we just print the error and continue sampling other candidates.


  # Filter out actions already in the expansion history of the parent assistant node
  unique_sampled_actions_info = []
  if node_to_expand_from: # Ensure parent exists
      print(f"Filtering sampled actions against expansion history of node {node_to_expand_from.node_id}...")
      for thought, action, output in sampled_actions_info:
          if action not in node_to_expand_from.expansion_history:
              unique_sampled_actions_info.append((thought, action, output))
              print(f"  - Added unique action: {action}")
          else:
              print(f"  - Skipping duplicate action already in history: {action}")

  # If no unique candidates, we should signal to backtrack
  if not unique_sampled_actions_info:
      # print("No unique expansion candidates found. Signaling backtracking.")
      return {"current_node": None, "node_stack": state["node_stack"]} # Return state to trigger backtrack edge

  # Prepare unique actions for critique
  actions_for_critique = [info[1] for info in unique_sampled_actions_info] # List of unique action strings

  selected_index, critic_prompt, critic_response = select_expansion( current_node, actions_for_critique)

  selected_thought, selected_action, selected_raw_output = unique_sampled_actions_info[selected_index]

  # Add the selected action string to the expansion history of the node being expanded from
  # This prevents trying this exact action again from this node in future expansions
  node_to_expand_from.expansion_history.append(selected_action)
  # print(f"Added selected action to expansion history of node {node_to_expand_from.node_id}: {selected_action}")

  assistant_node = append_history(
    {
        "role": "assistant",
        "content": output,
        "thought": thought,
        "action": action,
        "agent": "SWE-agent", # fixme: agent_name is harcoded
        "critic_prompt": critic_prompt,
        "critic_response": critic_response,
        "expansion_prompt": expansion_context,
    },
    parent_node=current_node
)

  observation = None # fixme ----------------------------------
  env_state = state.env.get_state(state) # fixme: undefined

  if len(current_node.children) == 1:

      if assistant_node.should_expand(DARS_CONFIG):
          assistant_node._action_expansion_limit[assistant_node.action.split()[0]] -= 1
          for _ in range(DARS_CONFIG.num_expansions-1):
              state.node_stack.append(assistant_node.parent)
          state.node_stack.sort(key=lambda x: x._depth)

  current_node = observation_state_to_node(
      last_node=assistant_node,
      state=env_state,
      observation=observation,
      search_term=search_term, # fixme: undefined
      codegraph_context=codegraph_context, # fixme: undefined
  )
  assert current_node.role == "user"
  return {
    "current_node": current_node,
    "node_stack": state.node_stack,
  }
