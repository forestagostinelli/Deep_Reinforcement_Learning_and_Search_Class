from typing import List, Tuple, Dict, Callable, Optional
from environments.environment_abstract import Environment, State
from heapq import heappush, heappop


class Node:
    def __init__(self, state: State, path_cost: float, parent_action: Optional[int], parent):
        """
        @param state: state
        @param path_cost: the cost to get from the start node to this node
        @param parent_action: the action that the parent node took to get to this node
        @param parent: the parent node
        """

        self.state: State = state
        self.path_cost: float = path_cost

        self.parent_action: Optional[int] = parent_action
        self.parent: Optional[Node] = parent

    def __hash__(self):
        return self.state.__hash__()

    def __eq__(self, other):
        return self.state == other.state

    def __lt__(self, other):
        return False


def astar(state: State, env: Environment, heuristic_fn: Callable, weight_g: float, weight_h: float) -> List[int]:
    """
    @param state: the state to be solved
    @param env: the environment
    @param heuristic_fn: the heuristic function
    @param weight_g: the weight for the path cost
    @param weight_h: the weight for the heuristic function

    @return: the list of actions to take find a path to the solution
    """
    # intialization
    open_queue: List[Tuple[float, Node]] = []
    closed: Dict[Node, float] = dict()

    root_node: Node = Node(state, 0.0, None, None)

    heur: float = heuristic_fn([root_node.state])[0]
    heappush(open_queue, (weight_g * root_node.path_cost + weight_h * heur, root_node))
    closed[root_node] = root_node.path_cost

    while len(open_queue) > 0:
        # pop node
        popped_node: Node = heappop(open_queue)[1]

    return []
