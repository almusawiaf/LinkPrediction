
import networkx as nx
import math

N = nx.Graph()
N.add_edges_from([(4,1),(4,2),(4,3),(4,5),(4,6),(5,6),(5,7),(5,8),(5,11),(6,12),(7,9),(7,10)])

def main(G):
    print("x \t y \t f \t cn \t aa \t ra \t pa \t ja \t sa \t so \t hpi \t hdi \t llhn \t car")
    for x in G.nodes:
        for y in G.nodes:
            f = 0
            if G.has_edge(x,y):
                f = 1
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
            print("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(x,y,f,cn,aa,ra,pa,ja,sa,so,hpi,hdi,llhn,car))


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
        print (L)
        if L != 0 :
            sum = sum + (1/L)
    return sum 

def RA(G, x, y):
    ax = set(G.neighbors(x))
    ay = set(G.neighbors(y))
    sum = 0 
    for z in (ax.intersection(ay)):
        sum = sum + (1/len(list(G.neighbors(z))))

    
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

    