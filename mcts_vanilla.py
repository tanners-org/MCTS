from mcts_node import MCTSNode
from p2_t3 import Board
from random import choice
from math import sqrt, log

num_nodes = 1000
explore_faction = 2

def traverse_nodes(node: MCTSNode, board: Board, state, bot_identity: int):
    """ Traverses the tree to find the best expandable node or a terminal node.

    Args:
        node:       A tree node from which the search is traversing.
        board:      The game setup.
        state:      The state of the game.
        bot_identity:   The bot's identity, either 1 or 2

    Returns:
        node: The best expandable or terminal node.
        state: The state associated with that node.

    """
    # Start traversing the tree
    while not board.is_ended(state):
        # If there are untried actions, consider all untried actions and select the best node based on UCB
        if len(node.untried_actions) > 0:
            # Find the best expandable node (one with untried actions) by comparing UCB values
            best_node = max(
                (n for n in node.child_nodes.values() if len(n.untried_actions) > 0), 
                key=lambda n: ucb(n, bot_identity), 
                default=None
            )
            if best_node:
                return best_node

        # If there are no untried actions, select the child node with the best UCB value
        if node.child_nodes:
            node = max(node.child_nodes.values(), key=lambda n: ucb(n, bot_identity))
            state = board.next_state(state, node.parent_action)  # Update the state as we traverse
        else:
            # If no child nodes and no untried actions, return the current node
            return node

    return node

def expand_leaf(node: MCTSNode, board: Board, state):
    """ Adds a new leaf to the tree by creating a new child node for the given node (if it is non-terminal).

    Args:
        node:   The node for which a child will be added.
        board:  The game setup.
        state:  The state of the game.

    Returns:
        node: The added child node
        state: The state associated with that node

    """
    # Select a random untried action
    action = choice(node.untried_actions)
    node.untried_actions.remove(action)  # Remove the chosen action from the list of untried actions
    new_state = board.next_state(state, action)  # Update the state
    new_child_node = MCTSNode(  # Create a new child node
        parent=node,
        parent_action=action,
        action_list=board.legal_actions(new_state))
    node.child_nodes[action] = new_child_node  # Add the new child node to the parent node
    return new_child_node, new_state  # Return the new child node



def rollout(board: Board, state):
    """ Given the state of the game, the rollout plays out the remainder randomly.

    Args:
        board:  The game setup.
        state:  The state of the game.
    
    Returns:
        state: The terminal game state

    """
    while not board.is_ended(state):
        legal_moves = board.legal_actions(state) # find legal moves
        random_move = choice(legal_moves) # select a random move
        state = board.next_state(state, random_move) # update the state

    return state


def backpropagate(node: MCTSNode | None, won: bool):
    """Navigates the tree from a leaf node to the root, updating the win and visit count of each node along the path.
    
    Args:
        node: A leaf node.
        won: A boolean indicator of whether the bot won (True) or lost (False).
    """
    while node is not None:
        node.visits += 1  # Increment visit count for this node
        
        # If it's the bot's turn, increment wins if won
        # If it's the opponent's turn, invert the result (i.e., lost for bot means won for opponent)
        if won:
            node.wins += 1  # Bot won, increment win counter
        else:
            # If the bot lost, we need to account for the opponent's win (invert the win count)
            node.wins -= 1
        
        # Move up to the parent node
        node = node.parent


def ucb(node: MCTSNode, is_opponent: bool):
    """ Calculates the UCB value for the given node from the perspective of the bot.
    
    Args:
        node:   A node.
        is_opponent: A boolean indicating whether or not the last action was performed by the MCTS bot.
        
    Returns:
        The value of the UCB function for the given node.
    """
    # If the node has not been visited, encourage exploration, but don't give max priority
    if node.visits == 0:
        exploration = sqrt(log(node.parent.visits + 1))  # A smaller exploration value
        return exploration

    exploitation = node.wins / node.visits
    
    # Adjust exploitation for the opponent's perspective
    if is_opponent:
        exploitation = 1 - exploitation  # Invert exploitation for opponent
    
    exploration = explore_faction * sqrt(log(node.parent.visits) / node.visits)
    return exploitation + exploration



def get_best_action(root_node: MCTSNode):
    """ Selects the best action from the root node in the MCTS tree

    Args:
        root_node:   The root node
    Returns:
        action: The best action from the root node
    
    """
    best_action = max(root_node.child_nodes.values(), key=lambda n: n.wins / n.visits) # find best action using winrate
    return best_action.parent_action

def is_win(board: Board, state, identity_of_bot: int):
    # checks if state is a win state for identity_of_bot
    outcome = board.points_values(state)
    assert outcome is not None, "is_win was called on a non-terminal state"
    return outcome[identity_of_bot] == 1

def think(board: Board, current_state):
    """ Performs MCTS by sampling games and calling the appropriate functions to construct the game tree.

    Args:
        board:  The game setup.
        current_state:  The current state of the game.

    Returns:    The action to be taken from the current state

    """
    bot_identity = board.current_player(current_state)  # The bot's identity (1 or 2)
    # print("mcts_vanilla is", bot_identity)
    root_node = MCTSNode(parent=None, parent_action=None, action_list=board.legal_actions(current_state))

    # MCTS iterations
    for _ in range(num_nodes):
        state = current_state
        node = root_node

        # Selection: Traverse the tree to find a promising node
        node = traverse_nodes(node, board, state, bot_identity)

        # Expansion: Expand the selected node if it has untried actions
        if len(node.untried_actions) > 0:
            node, state = expand_leaf(node, board, state)
        
        # Simulation: Perform a random rollout from the new state
        terminal_state = rollout(board, state)

        # Backpropagation: Update the tree with the results of the rollout
        won = is_win(board, terminal_state, bot_identity)
        backpropagate(node, won)

    # Return the best action from the root node
    best_action = get_best_action(root_node)
    
    #print(f"MCTS_vanilla chose: {best_action}")
    return best_action