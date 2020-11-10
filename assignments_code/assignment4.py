from typing import List, Tuple, Set, Dict, Callable, Optional, Any
from environments.environment_abstract import Environment, State
from heapq import heappush, heappop
from torch import nn


class FullyConnectedModel(nn.Module):
    def _forward_unimplemented(self, *input_val: Any) -> None:
        pass

    def __init__(self, input_dim: int, layer_dims: List[int], layer_batch_norms: List[bool], layer_acts: List[str]):
        super().__init__()
        self.layers: nn.ModuleList[nn.ModuleList] = nn.ModuleList()

        # layers
        for layer_dim, batch_norm, act in zip(layer_dims, layer_batch_norms, layer_acts):
            module_list = nn.ModuleList()

            # linear
            module_list.append(nn.Linear(input_dim, layer_dim))

            # batch norm
            if batch_norm:
                module_list.append(nn.BatchNorm1d(layer_dim))

            # activation
            act = act.upper()
            if act == "RELU":
                module_list.append(nn.ReLU())
            elif act == "SIGMOID":
                module_list.append(nn.Sigmoid())
            elif act != "LINEAR":
                raise ValueError("Un-defined activation type %s" % act)

            self.layers.append(module_list)

            input_dim = layer_dim

    def forward(self, x):
        x = x.float()

        module_list: nn.ModuleList
        for module_list in self.layers:
            for module in module_list:
                x = module(x)

        return x


class Node:
    def __init__(self, state: State, path_cost: float, parent_action: Optional[int], parent):
        self.state: State = state
        self.path_cost: float = path_cost

        self.cost: Optional[float] = None

        self.parent_action: Optional[int] = parent_action
        self.parent: Optional[Node] = parent

    def __hash__(self):
        return self.state.__hash__()

    def __eq__(self, other):
        return self.state == other.state


OpenSetElem = Tuple[float, int, Node]


class AStar:

    def __init__(self, state: State, env: Environment, heuristic_fn: Callable, weight_g: float, weight_h: float):
        self.env: Environment = env
        self.weight_g: float = weight_g
        self.weight_h: float = weight_h

        self.open_set: Set[Node] = set()
        self.open_priority_queue: List[OpenSetElem] = []
        self.heappush_count: int = 0
        self.closed_dict: Dict[Node, Node] = dict()
        self.goal_node: Optional[Node] = None

        # compute cost
        root_node: Node = Node(state, 0.0, None, None)
        costs = self.compute_cost([root_node], heuristic_fn)

        # push to open
        self.push_to_open([root_node], costs)

    def push_to_open(self, nodes: List[Node], costs: List[float]):
        for node, cost in zip(nodes, costs):
            node.cost = cost
            heappush(self.open_priority_queue, (cost, self.heappush_count, node))
            self.open_set.add(node)
            self.heappush_count += 1

    def pop_from_open(self) -> Node:
        popped_node: Node = heappop(self.open_priority_queue)[2]
        self.open_set.remove(popped_node)
        self.closed_dict[popped_node] = popped_node

        if self.env.is_terminal(popped_node.state):
            self.goal_node = popped_node

        return popped_node

    def is_solved(self) -> bool:
        return self.goal_node is not None

    def get_next_state_and_transition_cost(self, state: State, action: int) -> Tuple[State, float]:
        rw, states_a, _ = self.env.state_action_dynamics(state, action)
        state: State = states_a[0]
        transition_cost: float = -rw

        return state, transition_cost

    def remove_in_open_or_closed(self, nodes: List[Node]) -> List[Node]:
        nodes_not_removed: List[Node] = []

        for node in nodes:
            if node in self.open_set:
                continue

            if node in self.closed_dict:
                node_closed = self.closed_dict[node]
                if node_closed.path_cost > node.path_cost:
                    self.closed_dict.pop(node_closed)
                else:
                    continue

            nodes_not_removed.append(node)

        return nodes_not_removed

    def expand_node(self, parent_node: Node) -> List[Node]:

        return child_nodes

    def compute_cost(self, nodes: List[Node], heuristic_fn: Callable) -> List[float]:

        return costs

    def get_soln(self, node: Node) -> List[int]:

        return actions

    def step(self, heuristic_fn: Callable):
        # Pop from open
        popped_node: Node = self.pop_from_open()

        # Expand nodes
        nodes_c: List[Node] = self.expand_node(popped_node)

        # Check if children are in closed
        nodes_c = self.remove_in_open_or_closed(nodes_c)

        # Get heuristic of children
        if len(nodes_c) > 0:
            costs = self.compute_cost(nodes_c, heuristic_fn)

            # Add to open
            self.push_to_open(nodes_c, costs)
