import operator
from typing import Optional, Dict, Any, List
from typing_extensions import Annotated, TypedDict
from dataclasses import dataclass, field

import re
# from langchain_core.runnables import RunnableConfig
# from langgraph.constants import Send
# from langgraph.checkpoint.memory import MemorySaver


from prompt import (
    edit_expansion_prompt_template,
    append_expansion_prompt_template,
    submit_expansion_prompt_template,
    create_expansion_prompt_template,
    insert_expansion_prompt_template,
    critic_expansion_prompt_template,
    next_step_codegraph_template,
    next_step_no_output_template,
    next_step_template,
)

from config import DARS_CONFIG
from data_model import DARSNode, DARSState, AINode, UserNode

def last_n_history(history: List[Dict[str, Any]], n: int) ->  List[Dict[str, Any]]:
    if n <= 0:
        msg = "n must be a positive integer"
        raise ValueError(msg)
    new_history = list()
    user_messages = len([entry for entry in history if (entry["role"] == "user" and not entry.get("is_demo", False))])
    user_msg_idx = 0
    for entry in history:
        data = entry.copy()
        if data["role"] != "user":
            new_history.append(entry)
            continue
        if data.get("is_demo", False):
            new_history.append(entry)
            continue
        else:
            user_msg_idx += 1
        if user_msg_idx == 1 or user_msg_idx in range(user_messages - n + 1, user_messages + 1):
            new_history.append(entry)
        else:
            data["content"] = f'Old output omitted ({len(entry["content"].splitlines())} lines)'
            new_history.append(data)
    return new_history


# fixme: add demonstration!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def langgraph_node_to_message(node, n: int = 50, expansion_context: str = None) -> List[tuple[str, str]]:
    history: List[Dict[str, Any]] = last_n_history(local_history(node, expansion_context), n)
    output = []
    for entry in history:
        if entry["role"] == "user":
            output.append(("human", entry["content"]))
        elif entry["role"] == "assistant":
            output.append(("ai", entry["content"]))

    return output

def _backtrack_history(node: DARSNode, history: List[Dict[str, Any]]):
    if node is None:
        return

    _backtrack_history(node.parent, history)

    if node.is_demo:
        history.append({
            "agent": node.agent,
            "content": node.content,
            "is_demo": True,
            "role": node.role,
        })
    elif isinstance(node, AINode):
        history.append({
            "role": node.role,
            "content": node.content,
            "thought": node.thought,
            "action": node.action,
            "agent": node.agent
        })
    else:
        history.append({
            "role": node.role,
            "content": node.content,
            "agent": node.agent
        })

def local_history(node: DARSNode, expansion_context: str = None) -> list[dict[str, str]]:
    history = []

    _backtrack_history(node, history)
    history = [entry for entry in history if entry["agent"] == self.name] # fixme pls!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    base_history = DARS_CONFIG.history_processor(history) #fixme: history processor is not defined :(
    
    if expansion_context is None:
        return base_history
    
    info = ""
    if expansion_context:
        info = expansion_context
        last_message = base_history[-1]
        if last_message['role'] == 'user':
            last_message['content'] += f"\n\n{info}"
        else:
            base_history.append({"role": "user", "content": info})
    return base_history

def get_expansion_prompt(node: DARSNode) -> str:
    _command_handlers = {
        'edit': edit_expansion_prompt_template,
        'insert': insert_expansion_prompt_template,
        'append': append_expansion_prompt_template,
        'submit': submit_expansion_prompt_template,
        'create': create_expansion_prompt_template,
    }

    action = node.parent.action.split()[0]
    prompt_template = _command_handlers.get(action, '')
    if action != 'create' and action != 'submit':
        lookahead = DARS_CONFIG.n_lookahead
        context = ""
        patch = ""
        tmp = node.parent
        while lookahead:
            if tmp.action.split()[0] == "submit":
                patch = tmp.children[0].content
                if DARS_CONFIG.summarize_expansion_context:
                    break
            context += "ACTION: " + tmp.action + "\n"
            context += "OBERSERVATION: " + tmp.children[0].content[-10000:] + "\n\n"
            if not tmp.children or not tmp.children[0].children:
                break
            tmp = tmp.children[0].children[0]
            lookahead -= 1
        if DARS_CONFIG.summarize_expansion_context:
            context = summarize_content(context, patch)
        return prompt_template.format(action=node.parent.action, prev_traj=context)
    return prompt_template

def summarize_content(content: str, patch: str) -> str:

    out = DARS_CONFIG.model.query({"next_steps": content, "issue": DARS_CONFIG.issue}) # fixme=--------------------------------
    if patch != '': 
        out += "\n" + "Finally the model submitted the following changes:\n" + patch
    return out

def extract_best_action_index(response):
    pattern = r'<best_action_index>(\d+)</best_action_index>'
    match = re.search(pattern, response)
    if match:
        return int(match.group(1))
    raise ValueError("No best action index found in response")

def forward_model(last_node: DARSNode, expansion_context: str):
    query = local_history(last_node, expansion_context)

    #################################### fixme!!!!!!!!!!!!!!!!!  ##################################################################

    # last_message = query[-1]
    # self.logger.info(f"🤖 MODEL INPUT\n{last_message['content']}") 

    return DARS_CONFIG.model.query(query)


def select_expansion(node: DARSNode, actions: str) -> None: # fixme--------------------------------------------mypy fails
    assert node.role == "user"
    actions_context = ""
    for i, action in enumerate(actions):
        actions_context += f"Action {i}:\n{action}\n"
    critic_prompt = critic_expansion_prompt_template.replace("{actions}", f'\n{"".join(actions)}').replace("{previous_action}", node.children[0].action)
    response = forward_model(node, expansion_context=critic_prompt)
    try:
        action_index = extract_best_action_index(response)
    except Exception as e:
        # self.logger.warning(f"Error in selecting expansion: {e}")
        action_index = 0
    return action_index, critic_prompt, response


def observation_state_to_node(
        state: str, 
        observation: str, 
        search_term: str = "",
        codegraph_context: str = "",
        last_node: AINode = None,
) -> UserNode:
    open_file, working_dir = state
    if observation is None or observation.strip() == "":
        template = next_step_no_output_template.format(open_file=open_file, working_dir=working_dir)

    elif codegraph_context != "" and search_term != "":
        template = next_step_codegraph_template.format(
            search_term=search_term,
            codegraph_context=codegraph_context,
            open_file=open_file,
            working_dir=working_dir,
        )
    else:
        template = next_step_template.format(observation=observation, open_file=open_file, working_dir=working_dir)

    return append_history({
        "role": "user", 
        "content": template, 
        "agent": self.name,
        "codegraph_context": codegraph_context,
        "codegraph_keyword": search_term,
    }, parent_node=last_node)

def append_history(item: dict, parent_node: DARSNode = None, state: DARSState = None) -> DARSNode:
    # This needs to create a DARSNode and link it to the parent
    # It also needs to handle root node creation and node_id assignment
    # Assuming node_id assignment logic is handled elsewhere or passed in state

    # # Call hooks if they are in state
    # if state and state.get("hooks"):
    #     for hook in state["hooks"]:
    #         hook.on_query_message_added(**item) # Assuming hook method matches item keys

    new_node = DARSNode(
        role=item.get('role'),
        content=item.get('content'),
        agent=item.get('agent'),
        thought=item.get('thought'),
        action=item.get('action'),
        parent=None, # Set parent later
        is_demo=item.get('is_demo', False),
        is_terminal=item.get('is_terminal', False),
        # Copy other relevant attributes from item
        codegraph_keyword = item.get('codegraph_keyword'),
        codegraph_context = item.get('codegraph_context'),
        expansion_candidates = item.get('expansion_candidates'),
        critic_prompt = item.get('critic_prompt'),
        critic_response = item.get('critic_response'),
        expansion_prompt = item.get('expansion_prompt'),
        # expansion_history is initialized by default in DARSNode
        _action_expansion_limit = item.get('_action_expansion_limit', field(default_factory=dict)) # Copy if present
    )

    # Assign node_id - This logic needs to be managed globally, perhaps in state or a dedicated counter
    # if state and hasattr(state, 'node_count'): # Assuming node_count is in state
    #       new_node.node_id = state.node_count
    #       state['node_count'] += 1 # Increment global counter
    # elif parent_node: # Simple fallback if no global counter in state
    #       # This is not robust for unique IDs across branches, but works for basic tree structure
    #       new_node.node_id = hash(new_node) # Using hash is not ideal for unique IDs

    if state and state.get("root_node") is None:
        # If this is the first node, set it as the root
        if parent_node is not None:
              print("Warning: Parent node specified but root_node is None. Setting new node as root.")
        state["root_node"] = new_node
        # Initialize expansion limit for the root node's potential actions if needed
        # new_node._action_expansion_limit = state["config"].DARS.action_expansion_limit # If root can be expanded

    if parent_node:
        # Link the new node to its parent
        parent_node.add_child(new_node)

        if parent_node.is_terminal:
              new_node.is_terminal = True # Propagate terminal state

    # Return the newly created node
    return new_node