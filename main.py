import pandas as pd
import matplotlib.pyplot as plt
import re
from operator import itemgetter
import networkx as nx
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from collections import Counter
from scipy.stats import pearsonr


data = pd.read_csv('nytpopular.csv') 

entities = pd.read_csv('EntityData.csv')
ent = pd.read_csv('ID_Entity2.csv')

nodes = pd.read_csv('Nodes2.csv')
nodeData = pd.read_csv('NodeData.csv')

del_list = ['[', ']', "'", '"', ' ']

G = nx.Graph()



#####################################################
# -------- Graph construction and Analysis -------- #
#####################################################

#  1)
  
def plotHistograms(df, column, N):
    entities = []
    tags = []
    counts = {}
    for index, row in df.iterrows():
        if row[column] != '[]':
            temp = row[column].split(',')
            for ele in temp:
                for d in del_list:
                    ele = ele.replace(d, '')
                entities.append(ele)
                if ele not in tags:
                    tags.append(ele)
                    
    for tag in tags:
        counts[tag] = entities.count(tag)
        
    top_n = dict(sorted(counts.items(), key=itemgetter(1), reverse=True)[:N])

    plt.bar(list(top_n.keys()), top_n.values(), color='g')
    plt.show()

# Plotting degree distribution

def plotDegree(Network):
    degrees = [Network.degree(n) for n in Network.nodes()]
    nodes = [n for n in Network.nodes()]
    
    plt.hist(degrees)
    print(nodes)
    plt.show()
    
# Plotting top ten highest degree nodes and bag of phrase entities in terms of named-entities

def plotDegree(Network, dataframe, N):
    
    degrees = [Network.degree(n) for n in Network.nodes()]
    nodes = [n for n in Network.nodes()]
    nodeDegrees = {}
    
    for i in range(0, len(degrees)):
        nodeDegrees[nodes[i]] = degrees[i]
    
    top_ten = dict(sorted(nodeDegrees.items(), key=itemgetter(1), reverse=True)[:N])
    print(f"Top ten nodes based on degree")
    for node in top_ten:
        bag_of_phrases = dataframe['persons'].loc[dataframe['id'] == int(node)]
        print(f"Node id = {node}, Degree = {top_ten[node]}, Bag_of_phrases = {bag_of_phrases.item()}")  
    plt.hist(degrees)
    plt.show()
    
# Plotting betweenes degree and closeness centrality distribution and top ten nodes with bag of phrases

def plotBCdata(dataframe, nodeData, N):
    betweenes = []
    closeness = []
    
    for index, row in nodeData.iterrows():
        betweenes.append(row['centrality'])
        closeness.append(row['closeness'])
    
    plt.hist(betweenes)
    plt.show()
    plt.hist(closeness)
    plt.show()
    
    top_N_betweenes = nodeData.sort_values(by=['centrality'], ascending = False).head(N)
    top_N_closeness = nodeData.sort_values(by=['closeness'], ascending = False).head(N)
    
    print(f"")
    print(f"") 
    print(f"Top ten nodes based on Betweenes")
    print(f"") 
    for index, row in top_N_betweenes.iterrows():
        bag_of_phrases_person = dataframe['persons'].loc[dataframe['id'] == row['id']].item()
        bag_of_phrases_location = dataframe['locations'].loc[dataframe['id'] == row['id']].item()
        bag_of_phrases_organization = dataframe['organizations'].loc[dataframe['id'] == row['id']].item()
        bag_of_phrases = bag_of_phrases_location + bag_of_phrases_organization + bag_of_phrases_person
        print(f"Node id = {dataframe['id'].loc[dataframe['id'] == row['id']].item()}, Degree = {row['centrality'].item()}, Bag_of_phrases = {bag_of_phrases}")
        
    print(f"")
    print(f"")   
    print(f"Top ten nodes based on Closeness")
    print(f"") 
    for index, row in top_N_closeness.iterrows():
        bag_of_phrases_person = dataframe['persons'].loc[dataframe['id'] == row['id']].item()
        bag_of_phrases_location = dataframe['locations'].loc[dataframe['id'] == row['id']].item()
        bag_of_phrases_organization = dataframe['organizations'].loc[dataframe['id'] == row['id']].item()
        bag_of_phrases = bag_of_phrases_location + bag_of_phrases_organization + bag_of_phrases_person
        print(f"Node id = {dataframe['id'].loc[dataframe['id'] == row['id']].item()}, Degree = {row['closeness'].item()}, Bag_of_phrases = {bag_of_phrases}")
        
# Generating histogram of 10 bins on the values of the clustering coefficient
        
def plotClustering(nodeData):
    clustering = []
    
    for index, row in nodeData.iterrows():
        clustering.append(row['clustering'])
        
    plt.hist(clustering, bins=10, edgecolor="black")
    plt.show()


#####################################################
# --------    Graph attributes analysis    -------- #
##################################################### 

                         
def tokenPopularity(dataframe, originalData, types):
    
    for t in types:
        
        counts = {}
        for index, row in dataframe.iterrows():
            if row['persons'] != '[]':
                node_id = row['id']
                temp = row['persons'].split(',')
                for ele in temp:
                    for d in del_list:
                        ele = ele.replace(d, '')
                    val = originalData[t].loc[originalData['id'] == node_id].item()
                    if counts.get(str(ele)):
                        counts[str(ele)] = counts.get(str(ele)) + val
                    else:
                        counts[str(ele)] = val
                        
        top_five = dict(sorted(counts.items(), key=itemgetter(1), reverse=True)[:5])
        worst_five = dict(sorted(counts.items(), key=itemgetter(1), reverse=False)[:5])
        tag_count = []
        
        for x in top_five:
            new_row = {'token' : x, 'count' : top_five[x]}
            tag_count.append(new_row)
            
        for x in worst_five:
            new_row = {'token' : x, 'count' : worst_five[x]}
            tag_count.append(new_row)
        
        df = pd.DataFrame(tag_count)
        filename = t + '.csv'
        df.to_csv(filename, index=False)
        
        
def communities():
    G = read()
    sizes = []
    counts = []
    communities = nx.community.label_propagation_communities(G)
    for c in communities:
        if len(c) not in sizes and len(c) > 1:
            count = 0
            for x in communities:
                if len(x) == len(c):
                    count += 1
            sizes.append(len(c))
            counts.append(count) 
                           
    df = pd.DataFrame({'group_size': sizes, 'num_groups': counts})
    df.plot.barh(x='group_size', y='num_groups')
    plt.show()

def wordcloud_Communities(dataframe):
    
    G = read()
    communities = nx.community.label_propagation_communities(G)
    comSizes = {}
    
    for index, c in enumerate(communities):
        comSizes[index] = len(c)
    
    top_tree = dict(sorted(comSizes.items(), key=itemgetter(1), reverse=True)[:3])
    
    communityList = []
    
    for index, c in enumerate(communities):
        if index in list(top_tree.keys()):
            communityList.append(c)
    
    for index, com in enumerate(communityList):
        words = []
        for idd in com:
            data = dataframe['bag_of_phrases'].loc[dataframe['id'] == int(idd)]
            data = data.item()
            temp = data.split(',')
            for ele in temp:
                for d in del_list:
                    ele = ele.replace(d, '')
                words.append(ele)
        wordCount = Counter(words)
        path = 'community_' + str(index) + '.png'  
        wordcloud = WordCloud().fit_words(wordCount)
        wordcloud.to_file(path)
    
def pearsonCorrelations(dataframe):
    correlation_retweet, p_value_retweet = pearsonr(dataframe["retweet_count"],  dataframe["like_count"])
    correlation_reply, p_value_reply = pearsonr(dataframe["reply_count"],  dataframe["like_count"])
    print(f" Retweet_count - Like_count: Pearson correlation = {correlation_retweet} P-value = {p_value_retweet}")
    print(f" Reply_count - Like_count: Pearson correlation = {correlation_reply} P-value = {p_value_reply}")

    df = dataframe
  
    test = dataframe["bag_of_phrases"]
    counts = []
    
    for index, row in df.iterrows():
        row = row['bag_of_phrases'].split(',')
        counts.append(len(row))
    
    df['wordCount'] = counts
    
    correlation_rt, p_value_rt = pearsonr(df["retweet_count"],  df["wordCount"])
    correlation_rep, p_value_rep = pearsonr(df["reply_count"],  df["wordCount"])
    correlation_lik, p_value_lik = pearsonr(df["like_count"],  df["wordCount"])
    print(f" Retweet_count - num of tokens bag_of_phrases: Pearson correlation = {correlation_rt} P-value = {p_value_rt}")
    print(f" Reply_count - num of tokens bag_of_phrases: Pearson correlation = {correlation_rep} P-value = {p_value_rep}")
    print(f" Like_count - num of tokens bag_of_phrases: Pearson correlation = {correlation_lik} P-value = {p_value_lik}")


        
#####################################################
# --------    Simulation and randomness    -------- #
##################################################### 


def erdosNumbers(nodeData):
    G = read()
    index = nodeData.loc[nodeData['centrality'] == nodeData['centrality'].max()]
    index = index['id'].item()
    distances = nx.shortest_path_length(G, source=str(index))
    nodeDistance = []
    distanceList = []
    for dis in distances:
        nodeDistance.append({'node_id' : dis, 'distance' : distances[dis]})
        distanceList.append(distances[dis])
    
    df = pd.DataFrame(nodeDistance)
    df.to_csv('NodeDistanceData.csv', index=False)
    plt.hist(distanceList)
    plt.show()
    
#####################################################
# --------      Support functions          -------- #
##################################################### 


def read():
    file = open("test2.adjlist", "rb")
    G = nx.read_adjlist(file)
    return G


def showNetwork(G):
    nx.draw(G)
    plt.show()
    
          
def partOne():
    
    # -- Plotting histograms -- 
    plotHistograms(entities, 'persons', 10)   
    plotHistograms(entities, 'organizations', 10)
    plotHistograms(entities, 'locations', 10)
    # -- Plotting degree distribution and top ten nodes person tagged bag of phrases --
    plotDegree(read(), entities, 10)
    # -- Plotting betweenes degree and closeness centrality distribution and top ten nodes with bag of phrases --
    plotBCdata(entities, nodeData, 10)
    # -- Generating histogram of 10 bins on the values of the clustering coefficient --
    plotClustering(nodeData)

def partTwo():
    # -- calculating token popularity in case of retweet, reply and likecount -- 
    tokenPopularity(entities, data, ['retweet_count', 'reply_count', 'like_count'])
    # -- plot count of different size of communities -- 
    communities()
    # -- plot word map to tree biggest communities -- 
    wordcloud_Communities(data)
    # -- Calculates Pearson Correlations and P-value -- 
    pearsonCorrelations(data)

def partTree():
    # -- Calculates Erdos number for nodes in network --
    erdosNumbers(nodeData)
    











    







