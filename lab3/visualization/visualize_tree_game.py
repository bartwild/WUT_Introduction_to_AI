from anytree import Node
from anytree.exporter import UniqueDotExporter
from game import mini_max_with_tree


def create_children(tree_element, parrent_node):
    if tree_element["is_end"]:
        return
    else:
        for _, i in enumerate(tree_element["children"]):
            row = i["field"].row
            col = i["field"].col
            node = Node(str(i["best_score"][0][1]), parent=parrent_node, field=(row, col))
            create_children(i, node)


def visualize_tree_game(game, output_name):
    """
    Visualizes the game tree using graphviz and saves the output as a picture.

    Args:
    - game: an instance of the Game class representing the game to visualize
    - output_name: a string representing the name of the output file

    Returns:
    - None
    """
    tree_game_state = {}
    mini_max_with_tree(game.board, game.depth, tree_game_state)

    mainNode = Node(str(tree_game_state["best_score"][0][1]))

    create_children(tree_game_state, mainNode)

    def nodenamefunc(node):
        return '%s' % (str(node.score))

    def edgeattrfunc(node, child):
        return 'label="(%s,%s)"' % (str(child.field[0]), str(child.field[1]))

    UniqueDotExporter(mainNode, edgeattrfunc=edgeattrfunc).to_picture(output_name)
