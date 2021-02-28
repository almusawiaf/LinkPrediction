from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import networkx as nx
import math
import os 
from random import sample
import numpy as np
import statistics


# N = nx.Graph()
# N.add_edges_from([(4,1),(4,2),(4,3),(4,5),(4,6),(5,6),(5,7),(5,8),(5,11),(6,12),(7,9),(7,10)])
G = nx.karate_club_graph()
current_path = "D:/Documents/Research Projects/Complex Networks Researches/Link Prediction in dynamic networks/Python/"

def experiment():
    # G = read_graph()
    
    avg = "ag \t cn \t aa \t ra \t pa \t ja \t sa \t so \t hpi \t hdi \t llhn \t car \t AG_PA \n"
    std = ""
    for j in range(13):
        res = []
        for i in range(10):
            res.append(approach1(G, j))
        avg = avg + str(sum(res)/len(res)) + "\t"
        std = std + str(statistics.stdev(res)) + "\t"
    print(avg)
    print(std)
    
def read_graph():
    # U = input('u: undirected; *: directed')
    spath = "D:\Documents\Research Projects\Complex Networks Researches\Compaction Model\Intrinsic Model with approximating\Python Codes\\new Dataset"
    datasets = ["\\ba_1k_2k\\ba_1k_2k.txt",
                "\\bio-DM-LC\\bio-DM-LC.txt",
                "\\bio-diseasome\\bio-diseasome.txt",
                "\sociopatterns-hypertext\sociopatterns-hypertext.txt",
                "\\ba_1k_40k\\ba_1k_40k.txt",
                "\er_graph_1k_4k\er_graph_1k_4k.txt",
                "\er_graph_1k_6k\er_graph_1k_6k.txt",                   
                '\Ecoli\Ecoli.txt',
                '\\rt_assad\\rt_assad.txt',
                "\\bio-CE-LC\\bio-CE-LC.txt",
                "\\bio-yeast\\bio-yeast.txt",
                "\ca-CSphd\ca-CSphd.mtx",
                "\ca-GrQc\ca-GrQc.mtx",
                "\mammalia-voles-kcs-trapping\mammalia-voles-kcs-trapping.txt",
                "\socfb-Reed98\socfb-Reed98.mtx",
                "\socfb-Simmons81\socfb-Simmons81.mtx",
                "\\bio-CE-PG\\bio-CE-PG.txt",
                "\socfb-Haverford76\socfb-Haverford76.mtx",
                "\\bio-CE-CX\\bio-CE-CX.txt"]
    
    print("Select network index")
    for i in range(len(datasets)):
        print(i, datasets[i])
    g = int(input("Enter networks:  "))
    print ("you selected : ", datasets[g])
    G = nx.read_adjlist(spath + datasets[int(g)], create_using = nx.Graph(), nodetype = int)
    print(info(G))
    
    ppp = input("Press Enter to start")
    return G

def info(G):
    return "|V| = ", len(G.nodes()),"|E| = ",  len(G.edges())



def approach1(G, a):
    # find centrality of the network
    # divide the network G into E^T, E^P by randomly 80%, 20%
    # LP1(G^T)
    gT, gP = divide_80_20(G)
    gD = LP(gT, gP, a)    
    res = compare(gD, gP, G)
    tp_sum, fp_sum = 0,0
    for i in res:
        tp_sum = tp_sum + i[2]
        fp_sum = fp_sum + i[2]
    final = my_AUC(tp_sum, fp_sum, len(G.nodes()))
    # print("AUC = ", final)
    # print("-------------------------------------------")
    return final

def divide_80_20(G):
    E = list(G.edges())
    pT = round(sampling(80, len(E)))
    ET = sample(E, pT)
    EP = list(G.edges() - set(ET))
    
    gT = nx.Graph()
    gP = nx.Graph()
    
    gT.add_edges_from(ET)    
    gP.add_edges_from(EP)
    return gT, gP

def sampling(X, Total):
    return (X*Total/100)

def LP(gT, gP, a):
    # recieved G^T and return G' : gD 
    # a = selected LP algorithm
    gD = nx.Graph()
    Nodes = list(gT.nodes())
    res = []
    sum, n  = 0, 0
    for i in range(len(gT.nodes)-1):
        for j in range(i+1, len(gT.nodes)):
            x = Nodes[i]
            y = Nodes[j]
            # if not gT.has_edge(x, y):
            if a==0:
                Sxy = AG(gT, x, y)
            if a==1:
                Sxy = CN(gT, x, y)
            if a==2:
                Sxy = AA(gT, x, y)
            if a==3:
                Sxy = RA(gT, x, y)
            if a==4:
                Sxy = PA(gT, x, y)
            if a==5:
                Sxy = JA(gT, x, y)
            if a==6:
                Sxy = SA(gT, x, y)
            if a==7:
                Sxy = SO(gT, x, y)
            if a==8:
                Sxy = HPI(gT, x, y)
            if a==9:
                Sxy = HDI(gT, x, y)
            if a==10:
                Sxy = LLHN(gT, x, y)
            if a==11:
                Sxy = CAR(gT, x, y)
            if a==12:
                Sxy = AG2(gT, x, y)

            if gP.has_edge(x, y):
                sum = sum + Sxy
                n = n + 1
            res.append([x,y,Sxy])

    th = sum/n
    print(th)
    for x in res:
        if x[2]>=th:
            gD.add_edge(x[0], x[1])
    return gD

def threshold(R):
    t = []
    f = []
    for x in R:
        if x[0]==1:
            t.append(x[2])
        if x[1]==1:
            f.append(x[2])
    
def compare(gD, gP, G):
    """Return edge u,v and TP, FP"""
    Nodes = list(G.nodes())
    res = []
    for i in range(len(G.nodes)-2):
        for j in range(i+1, len(G.nodes)-1):
            u = Nodes[i]
            v = Nodes[j]
            TP = 0
            FP = 0
            if gP.has_edge(u, v) and gD.has_edge(u,v):
                TP = 1
            if gD.has_edge(u, v) and not G.has_edge(u,v):
                FP = 1 
            res.append([u,v,TP, FP])
    return res
            

def my_AUC(n1, n2, n):
    return (n1+0.5*n2)/n

def AUC(Data, alg, threshold):
    real_y = []
    predicted_y = []
    for X in Data:
        real_y.append(X[2])
        predicted_y.append(X[alg])
        # if X[alg]>threshold:
        #     predicted_y.append(1)
        # else:
        #     predicted_y.append(0)
    y_true = np.array(real_y)
    y_scores = np.array(predicted_y)
    for i in range(len(y_true)):
        print(y_true[i], y_scores[i])
    return roc_auc_score(y_true, y_scores)

def main(G):
    result = "ag \t cn \t aa \t ra \t pa \t ja \t sa \t so \t hpi \t hdi \t llhn \t car \n"
    # result = result + ("\n{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}".format(x,y,f,ag, cn,aa,ra,pa,ja,sa,so,hpi,hdi,llhn,car))
    # saving(Res)
    Nodes = list(G.nodes())
    Res = []    
    # Res.append(['x','y','f','ag', 'cn','aa','ra','pa','ja','sa','so','hpi','hdi','llhn','car'])
    #  Step 1: experiment
    for i in range(len(G.nodes)-1):
        for j in range(i+1, len(G.nodes)):
            x = Nodes[i]
            y = Nodes[j]
            f = 0
            if G.has_edge(x,y):
                f = 1
            ag = AG(G, x, y)
            cn = CN(G, x, y)
            aa = AA(G, x, y)
            ra = RA(G, x, y)
            pa = PA(G, x, y)
            ja = JA(G, x, y)
            sa = SA(G, x, y)
            so = SO(G, x, y)
            hpi = HPI(G, x, y)
            hdi = HDI(G, x, y)
            llhn = LLHN(G, x, y)
            car = CAR(G, x, y)
            Res.append([x,y,f,ag, cn,aa,ra,pa,ja,sa,so,hpi,hdi,llhn,car])
    # return (Res)
    #  Step 2: evaluation
    result = result + str(AUC(Res,3,1)) + "\t"
    result = result + str(AUC(Res,4,1)) + "\t"
    result = result + str(AUC(Res,5,1)) + "\t"
    result = result + str(AUC(Res,6,1)) + "\t"
    result = result + str(AUC(Res,7,1)) + "\t"
    result = result + str(AUC(Res,8,1)) + "\t"
    result = result + str(AUC(Res,9,1)) + "\t"
    result = result + str(AUC(Res,10,1)) + "\t"
    result = result + str(AUC(Res,11,1)) + "\t"
    result = result + str(AUC(Res,12,1)) + "\t"
    result = result + str(AUC(Res,13,1)) + "\t"
    result = result + str(AUC(Res,14,1)) + "\t"
    print(result)

def LP_ANN_Train(gT, gD):
    X, Y = Centrality_LP_Model(gT,1)
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)            
    clf.fit(X,Y)
    
    nX, nY = Centrality_LP_Model(gD, 1)
    resultY = list(clf.predict(nX))
    print(fit(nY, resultY))

def fit(Yrg, Yprd):
    tp = 0
    fp = 0
    for i in range(len(Yrg)):
        if Yrg[i] == 1:
            if Yprd[i] == 1:
                tp = tp + 1
            else:
                fp = fp + 1
    return tp, fp 

def Centrality_LP_Model(G, cent):
    C = get_centrality(G, cent)
    print(C)
    Nodes = list(G.nodes())
    X = []
    Y = []
    print(G.nodes())
    for i in range(len(G.nodes)-2):
        for j in range(i+1, len(G.nodes)-1):
            u = Nodes[i]
            v = Nodes[j]
            # print(i, j, u, v)
            Cu = C[i]
            Cv = C[j]
            tempX = [Cu,Cv]
            if (G.has_edge(u,v)):
                e_xy = 1
            else:
                e_xy = 0
            X.append(tempX)
            Y.append(e_xy)
    return X, Y
    
def get_centrality(G, cent):
    C = []
    if cent==1:
        C = nx.closeness_centrality(G)
    if cent==2:
        C = nx.eigenvector_centrality(G)
    if cent==3:
        C = nx.katz_centrality(G)
    if cent==4:
        C = nx.betweenness_centrality(G)
    return C        

def saving(T):
    title = input("Enter name of the file: ")
    text_file = open(current_path + title + ".txt", "w")
    text_file.write(T)
    text_file.close()

def AG(G, x, y):
    "Ahmad_Ghosh Model"
    deg = sum(d for n, d in G.degree()) / float(G.number_of_nodes())   
    tu = G.degree(x)/deg
    tv = G.degree(y)/deg
    potential  = abs(tu-tv)
    # print(tu, tv, potential)
    
    return (potential)

def AG2(G, x, y):
    "Ahmad_Ghosh Model"
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    sum1 = 0
    for xx in ax:
        sum1 = sum1 + G.degree(xx)        
    sum2 = 0
    for yy in ay:
        sum2 = sum2 + G.degree(yy)

    tu = G.degree(x)/(sum1/len(ax))
    tv = G.degree(y)/(sum2/len(ay))
    potential  = abs(tu - tv)
    # print(tu, tv, potential)
    
    return (potential)
        
def CN(G, x, y):
    'common neighbor'
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    return  abs(len(ax.intersection(ay)))

def AA(G,x,y):
    'Adamic-Adar index'
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    az = ax.intersection(ay)
    sum = 0
    for z in az:
        L = math.log(len(list(G.neighbors(z))))
        # print (L)
        if L != 0 :
            sum = sum + (1/L)
    return sum 

def RA(G, x, y):
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    sum = 0 
    for z in (ax.intersection(ay)):
        sum = sum + abs(1/len(list(G.neighbors(z))))
    return sum 

    
def PA(G, x, y):
    'Preferential Attachment'
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    return  len(ax)*len(ay)

def JA(G, x, y):
    'Jaccard Index'
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    return  len(ax.intersection(ay))/len(ax.union(ay))

def SA(G, x, y):
    'Salton Index'
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    return  len(ax.intersection(ay))/math.sqrt(len(ax)*len(ay))


def SO(G, x, y):
    'Sorensen Index'
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    return  2* len(ax.intersection(ay))/(len(ax)+len(ay))

def HPI(G, x, y):
    'Hub Pronoted Index'
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    return  len(ax.intersection(ay))/min(len(ax), len(ay))

def HDI(G, x, y):
    'Hub Depressed Index'
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    return  len(ax.intersection(ay))/max(len(ax), len(ay))

def LLHN(G, x, y):
    'Local Leicht-Homle-Newman Index'
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    return  len(ax.intersection(ay))/len(ax)*len(ay)

def CAR(G, x, y):
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    sum = 0 
    for z in (ax.intersection(ay)):
        az = G.neighbors(z)
        if len(list(az)) != 0:
            dom = len(ax.intersection(ay.intersection(set(G.neighbors(z)))))
            nom = len(list(G.neighbors(z)))
            sum = sum + (dom/nom)
    return sum