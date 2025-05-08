import operator
from typing import Optional, Dict, Any, List
from typing_extensions import Annotated, TypedDict

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
import uuid

from config import DARS_CONFIG

# @dataclass
# class DARSConfig:
#     """Simplified config placeholder."""
#     action_expansion_limit: Optional[Dict[str, int]] = field(default_factory=dict)
#     allowed_action_in_expansion: Optional[Dict[str, List[str]]] = field(default_factory=dict)
#     num_expansion_sampling: int = 3
#     # Add other config attributes as needed
# # --- End DARSConfig Placeholder ---

# --- Base DARSNode Class ---
@dataclass
class DARSNode:
    """Represents a base node in the DARS tree."""
    role: str # 'user' or 'assistant'
    content: str # Observation/State (user) or LLM raw output (assistant)
    agent: str # Name of the agent that generated this node

    is_demo: bool = field(default=False)
    is_terminal: bool = field(default=False)

    # Tree structure attributes
    parent: Optional['DARSNode'] = field(default=None, repr=False) # Parent node (for tree structure)
    children: List['DARSNode'] = field(default_factory=list) # Child nodes

    # Internal attributes
    _depth: int = field(default=0) # Depth in the tree, calculated in __post_init__
    node_id: str = field(default_factory=lambda: f"node-{uuid.uuid4()}") # Unique ID

    def __post_init__(self):
        """Calculate depth after initialization."""
        if self.parent is not None:
            self._depth = self.parent._depth + 1

    @property
    def depth(self) -> int:
        """Returns the depth of the node in the tree."""
        return self._depth

    def add_child(self, child: 'DARSNode'):
        """
        Adds a child node and sets its parent and depth.
        Expansion state propagation must be handled by the node creation logic
        in the LangGraph workflow, as it's type-specific.
        """
        child.parent = self
        # Child depth is calculated in the child's __post_init__ based on parent depth
        self.children.append(child)


# --- User Node Class ---
@dataclass
class UserNode(DARSNode):
    """Represents a user turn (environment observation/state)."""
    role: str = field(default="user", init=False) # Role is fixed as "user"

    # Attributes specific to UserNode (especially after search_repo)
    codegraph_keyword: Optional[str] = field(default=None)
    codegraph_context: Optional[str] = field(default=None)


# --- AI (Assistant) Node Class ---
@dataclass
class AINode(DARSNode):
    """Represents an assistant turn (LLM thought/action)."""
    role: str = field(default="assistant", init=False) # Role is fixed as "assistant"

    # Attributes specific to AINode
    thought: Optional[str] = field(default=None)
    action: Optional[str] = field(default=None) # The action string executed

    expansion_history: List[str] = field(default_factory=list) # History of action strings attempted from *this* AINode's action
    _action_expansion_limit: Dict[str, int] = field(default_factory=dict) # Remaining expansion budget for action types from *this* AINode's action

    # Expansion related attributes (set on AINode after expansion/critique)
    expansion_candidates: Optional[List[Dict[str, Any]]] = field(default=None) # Sampled alternative actions
    critic_prompt: Optional[str] = field(default=None) # Prompt used to critique candidates
    critic_response: Optional[str] = field(default=None) # Raw response from the critic LLM
    expansion_prompt: Optional[str] = field(default=None) # Prompt used to generate candidates


    def get_allowed_actions_in_expansion(self, dars_config: DARS_CONFIG) -> List[str]:
        """
        Determines the set of action types allowed for sampling alternative actions
        based on the expansion history *from this node's action*.
        This method is called on the Assistant node whose action is being expanded.
        """
        if not self.expansion_history:
            return (list(dars_config.action_expansion_limit.keys())
                    if dars_config.action_expansion_limit is not None
                    else [])

        if dars_config.allowed_action_in_expansion is None:
             return []

        # Logic remains the same, operating on self.expansion_history
        first_action_in_hist = self.expansion_history[0].split()[0] # Get action name from history string

        if first_action_in_hist not in dars_config.allowed_action_in_expansion:
            return []

        allowed_actions = set(dars_config.allowed_action_in_expansion[first_action_in_hist])

        for i, action_str in enumerate(self.expansion_history[1:]):
            action_type = action_str.split()[0]
            if action_type not in dars_config.allowed_action_in_expansion:
                return []
            actions_allowed_after_this = set(dars_config.allowed_action_in_expansion[action_type])
            allowed_actions.intersection_update(actions_allowed_after_this)
            if not allowed_actions:
                return []

        if dars_config.action_expansion_limit is not None:
            expandable_actions = set(dars_config.action_expansion_limit.keys())
            allowed_actions.intersection_update(expandable_actions)

        return list(allowed_actions)


    def should_expand(self, dars_config: DARS_CONFIG) -> bool:
        """
        Determines if *this* AINode's action is currently eligible for expansion.
        Checks if the action type is allowed by history and if the expansion budget remains.
        """
        if not self.action:
             return False # Cannot expand if there's no action

        action_type = self.action.split()[0]

        allowed_action_types = self.get_allowed_actions_in_expansion(dars_config)
        if not allowed_action_types or action_type not in allowed_action_types:
            print(f"Action type '{action_type}' is not in the allowed list based on history from this node.")
            return False

        if self._action_expansion_limit is not None:
            remaining_budget = self._action_expansion_limit.get(action_type, 0) # Use .get for safety
            return remaining_budget > 0
        else:
            return False # Cannot expand if limits are not tracked


class DARSState(TypedDict):
  current_node: DARSNode
  node_stack: List[DARSNode] #fixme-------------define list update
  root_node: DARSNode
  hooks: list #fixme-------------------
  env: Any #fixme----------------------
