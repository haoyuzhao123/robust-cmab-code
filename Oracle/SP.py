import networkx as nx
import numpy as np

def ShortestPath(P, params):
    ''' 
    return the shortest path list of the UCB weights graph G
    '''
    assert params["start"] != None and P.has_node(params["start"])
    assert params["end"] != None and P.has_node(params["end"])
    path_list = nx.shortest_path(P,params["start"],params["end"],weight="weight")
    # print(nx.shortest_path_length(P,params["start"],params["end"],weight="weight"))
    return path_list

def TargetPath(P):
    ''' 
    return the target path list of attack algorithm
    '''
    while True:
        s = np.random.randint(P.number_of_nodes())
        while P.has_node(s) == False:
            s = np.random.randint(P.number_of_nodes())
        path_list = [s]
        exists = {s: None}
        path_length = 50
        u = s
        sm = 0
        shortest_sm = 0
        tempG = nx.DiGraph()
        for x,y,z in P.edges(data="weight"):
            # print(x,y)
            tempG.add_edge(x,y,weight=1)
        for i in range(path_length):
            sample = []
            # print(P.out_degree(u))
            if P.out_degree(u) == 0:
                break
            for v in P.successors(u):
                if v not in exists:
                    sample.append(v)
            if len(sample) == 0:
                break
            v = np.random.choice(sample)
            exists[v] = None
            sm += P[u][v]["weight"] 
            tempG.add_edge(u,v,weight=P[u][v]["weight"])
            u = v
            path_list.append(v)
            if i > 0:
                t = v
                # shortest_sum = nx.shortest_path_length(P, s, t,weight="weight")
                shortest_path = nx.shortest_path(tempG, s, t,weight="weight")
                shortest_sm = nx.shortest_path_length(tempG, s, t,weight="weight")
                if sm-shortest_sm > 2:
                    print(sm, shortest_sm, sm - shortest_sm)
                if sm-shortest_sm > 3:
                    break
        t = path_list[-1]
        # print(path_list)
        # print(tempG[s][path_list[1]])
        # print(tempG[path_list[1]][t])
        shortest_path = nx.shortest_path(tempG, s, t,weight="weight")
        shortest_sm = nx.shortest_path_length(tempG, s, t,weight="weight")
        if sm-shortest_sm > 2:
            print(sm, shortest_sm, sm - shortest_sm)      
        if sm-shortest_sm > 3:
            break

    print("Target Path with weights",sm, path_list)
    print("shortest path with weights", shortest_sm, nx.shortest_path_length(P, s, t,weight="weight"), shortest_path)
    print("start",s,"end",t)
    return path_list, {"start": s, "end": t}

def TargetPath_Attackable(P):
    ''' 
    return the target path list of attack algorithm
    '''
    while True:
        s = np.random.randint(P.number_of_nodes())
        while P.has_node(s) == False:
            s = np.random.randint(P.number_of_nodes())
        path_list = [s]
        exists = {s: None}
        path_length = 50
        u = s
        sm = 0
        shortest_sm = 0
        tempG = nx.DiGraph()
        for x,y,z in P.edges(data="weight"):
            # print(x,y)
            tempG.add_edge(x,y,weight=1)
        for i in range(path_length):
            sample = []
            # print(P.out_degree(u))
            if P.out_degree(u) == 0:
                break
            for v in P.successors(u):
                if v not in exists:
                    sample.append(v)
            if len(sample) == 0:
                break
            v = np.random.choice(sample)
            exists[v] = None
            sm += P[u][v]["weight"] 
            tempG.add_edge(u,v,weight=P[u][v]["weight"])
            u = v
            path_list.append(v)
            if i > 0:
                t = v
                # shortest_sum = nx.shortest_path_length(P, s, t,weight="weight")
                shortest_wpath = nx.shortest_path(tempG, s, t, weight="weight")
                shortest_path = nx.shortest_path(P, s, t, weight="weight")
                shortest_sm = nx.shortest_path_length(tempG, s, t, weight="weight")
                if shortest_path == path_list:
                    continue
                if shortest_wpath==path_list:
                    print(sm, shortest_sm, sm - shortest_sm)
                if sm-shortest_sm <1e-6:
                    break
        t = path_list[-1]

        # print(path_list)
        # print(tempG[s][path_list[1]])
        # print(tempG[path_list[1]][t])
        shortest_wpath = nx.shortest_path(tempG, s, t, weight="weight")
        shortest_path = nx.shortest_path(P, s, t, weight="weight")
        shortest_sm = nx.shortest_path_length(tempG, s, t, weight="weight")
        if shortest_path == path_list:
            continue
        if shortest_wpath==path_list:
            print(sm, shortest_sm, sm - shortest_sm)
        if sm-shortest_sm <1e-6:
            break

    print("Target Path with weights",sm, path_list)
    print("shortest path with weights", shortest_sm, nx.shortest_path_length(P, s, t, weight="weight"), shortest_path)
    print("start",s,"end",t)
    return path_list, {"start": s, "end": t}

'''
def find_path(mid, u, v):
    if mid[u][v] != 0:
        return find_path(mid, u, mid[u][v]) + [mid[u][v]] + find_path(mid, mid[u][v], v)
    else:
        return []


def FLoyd(P, params):
    s = params["start"]
    t = params["end"]
    n = P.number_of_nodes()
    
    L = np.zeros((n+1,n+1)) - 1e6
    mid = np.zeros((n+1,n+1)).astype(int)
    for u, v, weight in P.edges(data="weight"):
        # print(u,v,weight)
        L[u][v] = weight
    for i in range(1, n+1):
        L[i][i] = 0
    for k in range(1,n+1):
        for i in range(1,n+1):
            for j in range(1,n+1):
                if L[i][k] + L[k][j] < L[i][j]:
                    L[i][j] = L[i][k] + L[k][j]
                    mid[i][j] = k
    path_list = [s]+find_path(mid, s, t)+[t]
    return path_list
'''