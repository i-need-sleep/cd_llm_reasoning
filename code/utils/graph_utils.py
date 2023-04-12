import networkx
import itertools

def complete_graph_from_list(L, create_using=None):
    G = networkx.empty_graph(len(L),create_using)
    if len(L)>1:
        if G.is_directed():
            edges = itertools.permutations(L,2)
        else:
            edges = itertools.combinations(L,2)
        G.add_edges_from(edges)
    return G