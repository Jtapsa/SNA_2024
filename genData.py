import pandas as pd
import matplotlib.pyplot as plt
import spacy
import re
from operator import itemgetter
import networkx as nx


data = pd.read_csv('nytpopular.csv') 
nlp = spacy.load('en_core_web_sm')
entities = pd.read_csv('EntityData.csv')
ent = pd.read_csv('ID_Entity2.csv')
nodes = pd.read_csv('Nodes2.csv')



del_list = ['[', ']', "'", '"', ' ']


# -- Extract the correct entity data from nytpopular.csv -- #

def extractEntities(data):
    subData = data.iloc[:,[0,6]]
    df = pd.DataFrame({'id': '',
    'persons': [],
    'organizations': [],
    'locations': []})
    
    for index, row in subData.iterrows():
        text =  row['bag_of_phrases']
        id_tweet = row['id']
        entities = nlp(text)
        persons = []
        organizations = []
        locations = []
        
        for ent in entities.ents:
            
            if ent.label_ == 'PERSON':
                persons.append(ent.text)
            
            if ent.label_ == 'ORG':
                organizations.append(ent.text)
            
            if ent.label_ == 'LOC':
                locations.append(ent.text)
        
        new_row = {'id': id_tweet, 'persons': persons, 'organizations': organizations, 'locations': locations}
        df.loc[index] = new_row
        df = df.reset_index(drop=True)
        print(index)
    
    df.to_csv('EntityData.csv', index=False)
    
    
# -- Extract all entity - tweet id pairs from EntityData.csv data -- #

def id_entities(dataframe, types):
    id_ent_list = []
    for idx, t in enumerate(types):
        column = t 
        for index, row in dataframe.iterrows():
            if row[column] != '[]':
                temp = row[column].split(',')
                for ele in temp:
                    for d in del_list:
                        ele = ele.replace(d, '')
                    new_row = {'id': int(row['id']), 'entity': ele, 'type': idx}
                    id_ent_list.append(new_row)
    df = pd.DataFrame(id_ent_list)
    df.to_csv('ID_Entity2.csv', index=False)
    

# -- Generate pairs of all nodes, which share same entity  -- #


def create_node_pairs(dataframe):
    tags = []
    
    # listing all unique enitites
    
    for index, row in dataframe.iterrows():
        if row['entity'] not in tags:
            tags.append(row['entity'])
            
    nodes = []
    
    for tag in tags:
        lista = dataframe.loc[dataframe['entity'] == tag]
        count = len(lista)
        print("current tag --> ", tag)
        print("Tweets with that tag --> ", count)
        tempp = []
        
        for idx, r in lista.iterrows():
            tempp.append(int(r['id']))
        
        tempp.sort()
        
        for idx, val in enumerate(tempp):
            print(f"cur idx --> {idx} of total --> {count}")
            for val2 in tempp[idx+1:]:
                new_row = {'id1': str(val), 'id2': str(val2), 'type' : str(r['type'])}
                nodes.append(new_row)
                             
    df = pd.DataFrame(nodes)
    df.to_csv('Nodes2.csv', index=False)
    
    
# -- Generate pairs of all nodes, which share same entity  -- #    
    
def generating_graph(nodes, edges):
    edges = edges.drop_duplicates()
    
    for index, row in nodes.iterrows():
        G.add_node(int(row['id']))
    
    for index, row in edges.iterrows():
        id1 = int(row['id1'])
        id2 = int(row['id2'])
        G.add_edge(id1, id2)
            
    nx.write_adjlist(G, "test2.adjlist")

# -- Node data from network  -- #  

def generateNodeData(G, dataframe):
    
    centrality = nx.betweenness_centrality(G)
    print("centrality check")
    closeness = nx.closeness_centrality(G)
    print("closeness check")
    clusterin = nx.clustering(G)
    print("clusterin check")
    
    nodeData = []
    
    for index, row in dataframe.iterrows():
        id = int(row['id'])
        new_row = {'id': id, 
                   'centrality': centrality.get(str(id)), 
                   'closeness': closeness.get(str(id)),
                   'clustering': clusterin.get(str(id))}
        
        nodeData.append(new_row)
        
    df = pd.DataFrame(nodeData)
    df.to_csv('NodeData.csv', index=False)
    

# -- Read saved Network  -- # 
    
def readNetwork():
    file = open("test2.adjlist", "rb")
    G = nx.read_adjlist(file)
    return G
    
    
    
def init():
    extractEntities(data)
    id_entities(entities, ['locations', 'persons', 'organizations'])
    create_node_pairs(ent)
    generating_graph(entities, nodes)
    generateNodeData(readNetwork(), entities)  

    
if __name__=="__main__": 
    #init()
    temp()