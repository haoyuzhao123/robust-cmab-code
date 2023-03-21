from optparse import OptionGroup
import networkx as nx
import numpy as np

def SpanningTree(P, params):
    ''' 
    return the spanning tree list of the UCB weights graph G
    '''
    assert params != None
    # print(P)
    path = nx.minimum_spanning_tree(P, weight="weight")

    # print(path.edges())
    # print(nx.shortest_path_length(P,params["start"],params["end"],weight="weight"))
    return path.edges()

def TargetST_Unattackable(P):
    '''
    generate the ST which has only one different edge with MST, and let the weight gap be the most.
    '''
    tempP = nx.Graph()
    for (u,v) in P.edges():
        tempP.add_edge(u, v, weight=-P[u][v]['weight'])
    target_tree = nx.minimum_spanning_tree(tempP, weight="weight")
    opt_tree = nx.minimum_spanning_tree(P, weight="weight")

    print("target edges",nx.number_of_edges(target_tree))
    print("opt edges",nx.number_of_edges(opt_tree))
    overlap = 0
    opt_weight = 0
    target_weight = 0
    for (u,v) in opt_tree.edges():
        if target_tree.has_edge(u,v):
            overlap += 1
        opt_weight += P[u][v]["weight"]
    for (u,v) in target_tree.edges():
        target_weight += P[u][v]["weight"]
    print("overlap",overlap)
    print("opt weight",opt_weight,"target weight",target_weight)
    return target_tree.edges(), {}


def find_min_edge_on_path(tree, u, v):
    path = nx.shortest_path(tree,u,v,weight='weight')
    if len(path)<2:
        print(path,u,v)
    ru,rv = path[0],path[1]
    for i in range(2,len(path)):
        u,v = path[i-1],path[i]
        if tree[u][v]['weight']<tree[ru][rv]['weight']:
            ru,rv = u,v
    return ru,rv

def find_max_edge_on_path(tree, u, v):
    path = nx.shortest_path(tree,u,v,weight='weight')
    if len(path)<2:
        print(path,u,v)
    ru,rv = path[0],path[1]
    for i in range(2,len(path)):
        u,v = path[i-1],path[i]
        if tree[u][v]['weight']>tree[ru][rv]['weight']:
            ru,rv = u,v
    return ru,rv

def TargetST_Attackable(P):
    '''
    generate the ST which has only one different edge with MST, and let the weight gap be the most.
    '''
    # tempP = nx.Graph()
    # for (u,v) in P.edges():
    #     tempP.add_edge(u, v, random_weight=np.random.rand())
    opt_tree = nx.minimum_spanning_tree(P, weight="weight")
    gap = 0 
    record = None
    print(P.number_of_edges(),opt_tree.number_of_edges())
    for (u,v) in P.edges():
        if opt_tree.has_edge(u,v) == False and u!=v:
            min_u,min_v = find_min_edge_on_path(opt_tree,u,v)    
            # print(P[u][v]['weight'] - P[min_u][min_v]['weight'])
            if P[u][v]['weight'] - P[min_u][min_v]['weight'] > gap:
                gap = P[u][v]['weight'] - P[min_u][min_v]['weight']
                record = (u,v,min_u,min_v)
    target_tree = opt_tree.copy()
    if record == None:
        print("NO other tree except MST")
        raise ValueError
    (u,v,min_u,min_v) = record
    target_tree.remove_edge(min_u,min_v)
    target_tree.add_edge(u,v,weight=P[u][v]["weight"])

    print("target edges",nx.number_of_edges(target_tree))
    print("opt edges",nx.number_of_edges(opt_tree))
    overlap = 0
    opt_weight = 0
    target_weight = 0
    for (u,v) in opt_tree.edges():
        if target_tree.has_edge(u,v):
            overlap += 1
        opt_weight += P[u][v]["weight"]
    for (u,v) in target_tree.edges():
        target_weight += P[u][v]["weight"]
    print("overlap",overlap)
    print("opt weight",opt_weight,"target weight",target_weight)
    return target_tree.edges(), {}

def TargetST_Random(P):
    ''' 
    return the random target path list (u,v) of attack algorithm
    '''
    tempP = nx.Graph()
    for (u,v) in P.edges():
        tempP.add_edge(u, v, random_weight=np.random.rand())
    path = nx.minimum_spanning_tree(tempP, weight="random_weight")
    opt = nx.minimum_spanning_tree(P, weight="weight")
    # print(path.edges(),len(path.edges()))
    # print("target",nx.number_of_edges(path))
    # print("opt",nx.number_of_edges(opt))
    total =  nx.number_of_edges(path)
    overlap = 0
    opt_weight = 0
    target_weight = 0
    for (u,v) in opt.edges():
        if path.has_edge(u,v):
            overlap += 1
        else:
            print("opt", u,v,P[u][v]["weight"])
        opt_weight += P[u][v]["weight"]
    for (u,v) in path.edges():
        target_weight += P[u][v]["weight"]
        if opt.has_edge(u,v) == False:
            print("target", u,v,P[u][v]["weight"])
    print("total edges", total, "overlap",overlap)
    print("opt weight",opt_weight,"target weight",target_weight)
    return path.edges(), {}

def TargetST_second(P):
    '''
    generate the second minimal ST.
    '''
    opt_tree = nx.minimum_spanning_tree(P, weight="weight")
    gap = 1e10
    record = None
    print(P.number_of_edges(),opt_tree.number_of_edges())
    for (u,v) in P.edges():
        if opt_tree.has_edge(u,v) == False and u!=v:
            max_u,max_v = find_max_edge_on_path(opt_tree,u,v)    
            # print(P[u][v]['weight'] - P[min_u][min_v]['weight'])
            if P[u][v]['weight'] - P[max_u][max_v]['weight'] < gap:
                gap = P[u][v]['weight'] - P[max_u][max_v]['weight']
                record = (u,v,max_u,max_v)
    target_tree = opt_tree.copy()
    if record == None:
        print("NO other tree except MST")
        raise ValueError
    (u,v,max_u,max_v) = record
    target_tree.remove_edge(max_u, max_v)
    target_tree.add_edge(u,v,weight=P[u][v]["weight"])

    print("target edges",nx.number_of_edges(target_tree))
    print("opt edges",nx.number_of_edges(opt_tree))
    overlap = 0
    opt_weight = 0
    target_weight = 0
    for (u,v) in opt_tree.edges():
        if target_tree.has_edge(u,v):
            overlap += 1
        else:
            print(u,v,P[u][v]["weight"])
        opt_weight += P[u][v]["weight"]
    for (u,v) in target_tree.edges():
        target_weight += P[u][v]["weight"]
        if opt_tree.has_edge(u,v) == False:
            print(u,v,P[u][v]["weight"])
    print("overlap",overlap)
    print("opt weight",opt_weight,"target weight",target_weight)
    return target_tree.edges(), {}
    