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

def TargetPath_RandomWalk(P):
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
            u = v
            path_list.append(v)

        t = path_list[-1]
        # print(path_list)
        # print(tempG[s][path_list[1]])
        # print(tempG[path_list[1]][t])
        shortest_path = nx.shortest_path(P, s, t,weight="weight")
        if shortest_path != path_list:
            break
    
    target = []
    for i in range(len(path_list)):
        if i==0:
            u = path_list[i]
            continue
        v = path_list[i]
        target.append((u,v))
        u = v
    print("Target Path with weights", path_list)
    print("shortest path with weights", shortest_path)
    print("start",s,"end",t)
    return target, {"start": s, "end": t}


def TargetPath_RandomWeight(P):
    ''' 
    return the random target path list (u,v) of attack algorithm
    '''
    tempP = nx.DiGraph()
    for (u,v) in P.edges():
        tempP.add_edge(u, v, random_weight=np.random.rand())
    for _ in range(10000):
        s = np.random.randint(P.number_of_nodes())
        path_from_s = nx.shortest_path(P, source = s)
        for i in range(5):
            t = np.random.randint(P.number_of_nodes())
            if t in path_from_s:
                opt = path_from_s[t]
                path = nx.shortest_path(tempP, s, t, weight="random_weight")
                if path != opt:
                    return reshape_path_opt(P, path,opt, s, t)

    raise ValueError("can not find target path")

def TargetPath_RandomTri(P):
    ''' 
    return the random target path list (u,v) of attack algorithm
    '''
    tri_list=[]
    for (u,v) in P.edges():
        for k in P.nodes():
            if u!=k and v!=k:
                if P.has_edge(u,k) and P.has_edge(k,v):
                    tri_list.append((u,k,v))

    print("number of triangles",len(tri_list))
    # for (u,k,v) in tri_list:
    #     for pre_u in u.predecessor
    for _ in range(10000):
        s = np.random.randint(P.number_of_nodes())
        path_from_s = nx.shortest_path(P, source = s)
        for i in range(5):
            t = np.random.randint(P.number_of_nodes())
            if t in path_from_s:
                opt = path_from_s[t]
                path = nx.shortest_path(tempP, s, t, weight="random_weight")
                if path != opt:
                    return reshape_path_opt(P, path,opt, s, t)

    raise ValueError("can not find target path")

def reshape_path_opt(P, path,opt,s, t):
    opt_weight = 0
    target_weight = 0
    target = []
    for i in range(len(path)):
        if i==0:
            u = path[i]
            continue
        v = path[i]
        target.append((u,v))
        target_weight+=P[u][v]["weight"]
        u = v
    for i in range(len(opt)):
        if i==0:
            u = opt[i]
            continue
        v = opt[i]
        opt_weight+=P[u][v]["weight"]
        u = v
    print("Target Path with weights", target_weight, path)
    print("shortest path with weights", opt_weight, opt)
    print("start",s,"end",t)
    return target, {"start": s, "end": t}

    
def TargetPath_Unattackable(P):
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
                # print(sm, shortest_sm, sm - shortest_sm)
                if sm-shortest_sm > 0.5:
                    print(sm - shortest_sm, sm, shortest_sm, path_list, shortest_path)
                if sm-shortest_sm >= 0.5 and len(shortest_path)>2:
                    break
        t = path_list[-1]
        # print(path_list)
        # print(tempG[s][path_list[1]])
        # print(tempG[path_list[1]][t])
        shortest_path = nx.shortest_path(tempG, s, t,weight="weight")
        shortest_sm = nx.shortest_path_length(tempG, s, t,weight="weight")   
        if sm-shortest_sm >= 0.5 and len(shortest_path)>2:
            break
    
    target = []
    for i in range(len(path_list)):
        if i==0:
            u = path_list[i]
            continue
        v = path_list[i]
        target.append((u,v))
        u = v
    print("Target Path with weights",sm, path_list)
    print("shortest path with weights", shortest_sm, nx.shortest_path_length(P, s, t,weight="weight"), shortest_path)
    print("start",s,"end",t)
    return target, {"start": s, "end": t}

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
        
    target = []
    for i in range(len(path_list)):
        if i==0:
            u = path_list[i]
            continue
        v = path_list[i]
        target.append((u,v))
        u = v
    print("Target Path with weights",sm, path_list)
    print("shortest path with weights", shortest_sm, nx.shortest_path_length(P, s, t, weight="weight"), shortest_path)
    print("start",s,"end",t)
    return target, {"start": s, "end": t}

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

def TargetPath_Unattackable2(P):
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
                # print(sm, shortest_sm, sm - shortest_sm)
                if sm-shortest_sm > 0.5:
                    print(sm - shortest_sm, sm, shortest_sm, path_list, shortest_path)
                if sm-shortest_sm >= 0.5:
                    break
        t = path_list[-1]
        # print(path_list)
        # print(tempG[s][path_list[1]])
        # print(tempG[path_list[1]][t])
        shortest_path = nx.shortest_path(tempG, s, t,weight="weight")
        shortest_sm = nx.shortest_path_length(tempG, s, t,weight="weight")   
        if sm-shortest_sm >= 0.5:
            break
    
    target = []
    for i in range(len(path_list)):
        if i==0:
            u = path_list[i]
            continue
        v = path_list[i]
        target.append((u,v))
        u = v
    print("Target Path with weights",sm, path_list)
    print("shortest path with weights", shortest_sm, nx.shortest_path_length(P, s, t,weight="weight"), shortest_path)
    print("start",s,"end",t)
    return target, {"start": s, "end": t}