import networkx as nx
import matplotlib.pyplot as plt
import json

import numpy as np
from keras.layers import Activation, Dense, Input, initializers 
from keras.models import model_from_json,  load_model, Sequential
from keras import optimizers

def createGraph(model):

    """createGraph:
    
    Créée un graph networkX à partir d'un modèle contenant les poids et les biais d'un réseau de neuronnes
    
    paramètres:
        -modelFile: fichier du modèle à transcrire
    
    renvoie:
        graph créé

    """
    #CHARGEMENT DU MODELE

    if type(model) == str:
        model = load_model(model)
    weights  = model.get_weights()
    arch = model.to_json()

    
    #CREATION DU GRAPHE
    
    graph = nx.Graph()

    #PARAMETRES - NOMBRE DE COUCHES

    graph.graph['nbCouches'] = int(len(weights) / 2 + 1)#nombre de couches (entrées et sorties comprises)

    #FONCTIONS D'ACTIVATIONS
    
    graph.graph['fonctions'] = []
    arch = json.loads(arch)
    for i in range(graph.graph['nbCouches'] - 1):
        graph.graph['fonctions'].append(arch['config'][i]['config']['activation'])
    
    #PARAMETRES - NOMBRE DE NEURONES PAR COUCHES

    graph.graph['nbNodes'] = [] #liste du nombre de neurone par couche
    graph.graph['maxNeuronnesCouche'] = 0 #nombre de neurones dans la plus grosse couche

    for i in range(graph.graph['nbCouches']):
        if i > 0:
            graph.graph['nbNodes'].append(int(len(weights[i*2-1])))
        else:
            graph.graph['nbNodes'].append(int(len(weights[0])))
        if graph.graph['nbNodes'][i] > graph.graph['maxNeuronnesCouche']:
            graph.graph['maxNeuronnesCouche'] = graph.graph['nbNodes'][i]

    #RECHERCHE DU PLUS GROS BIAIS

    graph.graph['bestBiais'] = 0#Biais avec la valeur absolue la plus élevée

    for i in weights[1::2]:
        for j in i:
            if abs(j) > graph.graph['bestBiais']:
                graph.graph['bestBiais'] = abs(j)

    #AJOUT DE NODES
    for i in range(graph.graph['nbCouches']):
        for j in range(graph.graph['nbNodes'][i]):
            
            graph.add_node(i * graph.graph['maxNeuronnesCouche'] + j)

    #AJOUT DES EDGES
    for i in range(graph.graph['nbCouches']):
        for j in range(graph.graph['nbNodes'][i]):
            if i < graph.graph['nbCouches'] - 1:
                for k in range(graph.graph['nbNodes'][i+1]):
                    poids = weights[2 * i][j][k]#Détermination de leur poids
                    graph.add_edge(i * graph.graph['maxNeuronnesCouche'] + j, (i+1) * graph.graph['maxNeuronnesCouche'] + k)
                    graph[i * graph.graph['maxNeuronnesCouche'] + j][(i+1) * graph.graph['maxNeuronnesCouche'] + k]['poids'] = poids

    graph.graph['upperPoids'] = np.max(list(nx.get_edge_attributes(graph, 'poids').values()))#Poids le plus élevé
    graph.graph['lowerPoids'] = np.min(list(nx.get_edge_attributes(graph, 'poids').values()))#Poids le plus bas

    if abs(graph.graph['upperPoids']) > abs(graph.graph['lowerPoids']):
        graph.graph['bestPoids'] = abs(graph.graph['upperPoids'])
    else:
        graph.graph['bestPoids'] = abs(graph.graph['lowerPoids'])#Poids avec l'absolu le plus élevé

    #DETERMINATION DE L'IMPORTANCE DE CHAQUE NODE

    for i in range(graph.graph['nbCouches']):
        for j in range(graph.graph['nbNodes'][i]):
            graph.nodes[i * graph.graph['maxNeuronnesCouche'] + j]['importance'] = 0
            if i < graph.graph['nbCouches'] - 1:
                for k in range(graph.graph['nbNodes'][i+1]):
                    graph.nodes[i * graph.graph['maxNeuronnesCouche'] + j]['importance'] += abs(graph[i * graph.graph['maxNeuronnesCouche'] + j][(i+1) * graph.graph['maxNeuronnesCouche'] + k]['poids']) / graph.graph['nbNodes'][i+1]

    #CREATION D'UN DICTIONNAIRE POUR LES POSITIONS DES NODES

    graph.graph['posNodes'] = dict()
    for i in range(graph.graph['nbCouches']):
        for j in range(graph.graph['nbNodes'][i]):
            graph.graph['posNodes'][i * graph.graph['maxNeuronnesCouche'] + j] = (i, j + (graph.graph['maxNeuronnesCouche'] - graph.graph['nbNodes'][i]) / 2)

    #DETERMINATION DES ATTRIBUTS VISUELS DES EDGES

    for i in range(graph.graph['nbCouches']):
        for j in range(graph.graph['nbNodes'][i]):
            if i < graph.graph['nbCouches'] - 1:
                for k in range(graph.graph['nbNodes'][i+1]):
                    #Epaisseur de l'edge en fonction du poids de la liaison
                    poids = graph[i * graph.graph['maxNeuronnesCouche'] + j][(i+1) * graph.graph['maxNeuronnesCouche'] + k]['poids']
                    graph[i * graph.graph['maxNeuronnesCouche'] + j][(i+1) * graph.graph['maxNeuronnesCouche'] + k]['epaisseur'] = abs(poids) / graph.graph['bestPoids'] * 30
            
    #Dictionnaire pour les couleurs
    
    graph.graph['couleurs'] = {}#Contient les couleurs de chaque fonction d'activation, en RGB
    
    graph.graph['couleurs']["input"]  = (1, 1, 1)
    graph.graph['couleurs']["linear"]  = (1, 1, 0)
    graph.graph['couleurs']["elu"]  = (1, 0, 0)
    graph.graph['couleurs']["relu"]  = (0.6, 0, 0)
    graph.graph['couleurs']["selu"]  = (1, 0.4, 0.4)
    graph.graph['couleurs']["tanh"]  = (0.5, 0.5, 0.5)
    graph.graph['couleurs']["sigmoid"]  = (0.1, 0.1, 0.8)
    graph.graph['couleurs']["hard_sigmoid"]  = (0, 0, 0.5)
    graph.graph['couleurs']["softmax"]  = (0, 1, 0)
    graph.graph['couleurs']["softplus"]  = (0.4, 1, 0.4)
    graph.graph['couleurs']["softsign"]  = (0, 0.6, 0)

    return graph

def show(graph, saveFile = "", dpi = 200, start = 0, legender = True, fullLegende = False, titre = ""):
    
    """Show:
    
    affiche un réseau de neurones organisé dans un graph networkX
    
    paramètres:
        -graph: réseau de neuronnes à aficher
        -saveFile(optionel, défaut = ""): fichier pour enregistrer l'image (.png de préférence). envoyer "" pour ne pas enregistrer
        -dpi(optionel, défaut = 200): qualité de l'eregistrement dans saveFile
        -start(optionel, défaut = 0): tranche de 1% de poids liaisons à laquelle on commence pour les afficher (1 affiche tout, 25 n'affiche pas les liaisons de poids inferieures à 1/4 du poids de la plus grosse)
        -legender(optionel, défaut = True): affichage ou non des légendes
        -fullLegende(optionel, défaut = False): affichage ou non des légendes inutiles (valable uniquement si legender = True)
        -titre(optionel, défaut = ""): titre à afficher mettez un '||' pour séparer titre / sous-titre 
        
    renvoie:
        image créée (par matplotlib (plt))
        
    """
    
    #DESSIN DES NODES
    
    plt.clf()
    plt.cla()

    plt.figure(figsize=(5 * graph.graph['nbCouches'] + 5 * legender,1.5 * graph.graph['maxNeuronnesCouche']))

    tailleCentreNodes = 100 + 100 / graph.graph['maxNeuronnesCouche'] #Taille de la partie interieure des neuronnes
    
    for node in graph.nodes():
        i = int (node / graph.graph['maxNeuronnesCouche'])#Numéro de la couche déterminée par le numéro général du neuronne
        j = node % graph.graph['maxNeuronnesCouche']#Numéro du neuronne dans la couche  déterminée par le numéro général du neuronne

        #Dessin de l'importance
        
        if i > 0:
            color = graph.graph['couleurs'][graph.graph['fonctions'][i-1]]
        else:
            color = (1, 1, 1)
        
        if i == graph.graph['nbCouches'] - 1:
            nx.draw_networkx_nodes(graph, graph.graph['posNodes'], nodelist = [node], node_size = 200 + tailleCentreNodes + 4000, node_color = 'k')
            nx.draw_networkx_nodes(graph, graph.graph['posNodes'], nodelist = [node], node_size = tailleCentreNodes + 4000, node_color = color)
        else:
            importance = graph.nodes[i * graph.graph['maxNeuronnesCouche'] + j]['importance']
            nx.draw_networkx_nodes(graph, graph.graph['posNodes'], nodelist = [node], node_size = 200 + tailleCentreNodes + importance * 10000, node_color = 'k')
            nx.draw_networkx_nodes(graph, graph.graph['posNodes'], nodelist = [node], node_size = tailleCentreNodes + importance * 10000, node_color = color)

    nx.draw_networkx_nodes(graph, graph.graph['posNodes'], node_size = tailleCentreNodes, node_color = 'k')
    
    #DESSIN DES EDGES

    for i in range(start, 101):
        for edge in graph.edges():
            poids = graph[edge[0]][edge[1]]['poids']
            if abs(poids) > graph.graph['bestPoids'] * (i-1) / 100 and abs(poids) <= graph.graph['bestPoids'] * i / 100:
                epaisseur = graph[edge[0]][edge[1]]['epaisseur']
                nx.draw_networkx_edges(graph, graph.graph['posNodes'], edgelist = [edge], edge_color = 'k', width = 1 + epaisseur)
                if poids > 0:
                    nx.draw_networkx_edges(graph, graph.graph['posNodes'], edgelist = [edge], edge_cmap = plt.cm.Greens, edge_vmin = 0, edge_vmax = graph.graph['bestPoids'], edge_color = [abs(poids)], width = epaisseur)
                else:
                    nx.draw_networkx_edges(graph, graph.graph['posNodes'], edgelist = [edge], edge_cmap = plt.cm.Blues, edge_vmin = 0, edge_vmax = graph.graph['bestPoids'], edge_color = [abs(poids)], width = epaisseur)
     
    #TITRE
    if titre != "":
        titre = titre.split('|')
        if len(titre) == 1:
             plt.text(graph.graph['nbCouches'] / 2, graph.graph['maxNeuronnesCouche'] + 1, titre[0], size = 64, ha = 'center', va = "top")
        else:
            plt.text(graph.graph['nbCouches'] / 2, graph.graph['maxNeuronnesCouche'] + 1, titre[0], size = 64, ha = 'center', va = 'top')
            plt.text(graph.graph['nbCouches'] / 2, graph.graph['maxNeuronnesCouche'] + 0.25, titre[1], size = 48, ha = 'center', va = 'top')

    #LEGENDE
    
    if legender:
    
        legende = nx.Graph()
        
        posLegendeNodes = {}
        
        i = 1

        legende.add_node('input')
        posLegendeNodes['input'] = (graph.graph['nbCouches'], graph.graph['maxNeuronnesCouche'])
        plt.text(graph.graph['nbCouches'] - 0.1, graph.graph['maxNeuronnesCouche'], "input:", size = 32, ha = 'right', va = 'center')
        
        
        for key in graph.graph['couleurs']:
            if key in graph.graph['fonctions'] or fullLegende:
                legende.add_node(key)
                posLegendeNodes[key] = (graph.graph['nbCouches'], graph.graph['maxNeuronnesCouche'] - i)
                plt.text(graph.graph['nbCouches'] - 0.1, graph.graph['maxNeuronnesCouche'] - i, key + ":", size = 32, ha = 'right', va = 'center')
                i += 1

        for node in legende.nodes():
            color = graph.graph['couleurs'][node]
            nx.draw_networkx_nodes(legende, posLegendeNodes, nodelist = [node], node_size = 200 + tailleCentreNodes + 2000, node_color = 'k')
            nx.draw_networkx_nodes(legende, posLegendeNodes, nodelist = [node], node_size = tailleCentreNodes + 2000, node_color = color)

        nx.draw_networkx_nodes(legende, posLegendeNodes, node_size = tailleCentreNodes, node_color = 'k')
        
    #SAUVEGARDE
    
    if saveFile != "":
        plt.savefig(saveFile, dpi = dpi)
    
    return plt.gcf()

def test():

    model = Sequential()

    model.add(Dense(units=30, activation='relu', input_dim=20))
    model.add(Dense(units=10, activation='relu'))
    model.add(Dense(units=1, activation='linear'))

    graph = createGraph(model)
    
    graphImage = show (graph, start = 12, titre = "Exemple|Non entrainé")