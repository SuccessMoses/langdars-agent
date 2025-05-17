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


    def get_allowed_actions_in_expansion(self, dars_config: DARS_CONFIG):
        if not self.expansion_history:
            return (list(dars_config.action_expansion_limit.keys()) 
                    if dars_config.action_expansion_limit is not None 
                    else [])

        first_action = self.expansion_history[0]
        if first_action not in dars_config.allowed_action_in_expansion:
            return []
        
        allowed_actions = set(dars_config.allowed_action_in_expansion[first_action])
        
        for action in self.expansion_history[1:]:
            if action not in dars_config.allowed_action_in_expansion:
                return []
            allowed_actions.intersection_update(dars_config.allowed_action_in_expansion[action])
            if not allowed_actions:
                return []
        if dars_config.action_expansion_limit is not None:
            allowed_actions.intersection_update(dars_config.action_expansion_limit.keys())

        return list(allowed_actions)

    def should_expand(self, dars_config: DARS_CONFIG):
        allowed_actions = self.get_allowed_actions_in_expansion(dars_config)
        action = self.action.split()[0]
        if not allowed_actions or action not in allowed_actions:
            return False
        
        if self._action_expansion_limit is not None:
            return self._action_expansion_limit[action] > 0
        return False


class DARSState(TypedDict):
  current_node: DARSNode
  node_stack: List[DARSNode] #fixme-------------define list update
  root_node: DARSNode
  hooks: list #fixme-------------------
  env: Any #fixme----------------------
