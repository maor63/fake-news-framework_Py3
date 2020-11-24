
import gensim.models.doc2vec as doc
import os
import random
import networkx as nx
from itertools import chain


def arr2str(arr):
    return ' '.join(arr)


def randomWalkDegreeLabels(G, walkSize, iterations=10):
    nodes = list(G.nodes(data=False))
    random_walks = []
    for i in range(iterations):
        random.shuffle(nodes)
        random_walks += list(chain(*[random_walk_node(G, node, walkSize) for node in nodes]))
    return random_walks


def random_walk_node(G, curNode, walkSize):
    walkList = []
    for i in range(walkSize):
        walkList.append(G.node[curNode]['label'])
        candidates = list(G.neighbors(curNode))
        curNode = random.choice(candidates)
    return walkList


def getDegreeLabelledGraph(G, rangetoLabels):
    degreeDict = G.degree(G.nodes())
    labelDict = {}
    for node in degreeDict:
        val = degreeDict[node] / float(nx.number_of_nodes(G))
        labelDict[node] = inRange(rangetoLabels, val)
    # nx.set_node_attributes(G, labelDict, 'label')
    nx.set_node_attributes(G, values=labelDict, name='label')
    return G


def inRange(rangeDict, val):
    for key in rangeDict:
        if key[0] < val <= key[1]:
            return rangeDict[key]


def generateWalkFile(graphs, file_name, walkLength, alpha, random_walk_count):
    walkFile = open(file_name, 'w')
    indexToName = {}
    rangetoLabels = {(-0.01, 0.05): 'z', (0.05, 0.1): 'a', (0.1, 0.15): 'b', (0.15, 0.2): 'c', (0.2, 0.25): 'd',
                     (0.25, 0.5): 'e', (0.5, 0.75): 'f', (0.75, 1.0): 'g'}
    for index, graph in enumerate(graphs):
        print('\r generate graph embedding {}/{}'.format(str(index+1), len(graphs)), end='')
        subgraph = nx.Graph(graph)
        degreeGraph = getDegreeLabelledGraph(subgraph, rangetoLabels)
        subgraph.add_edges_from([(node, node) for node in subgraph.nodes()])
        degreeWalk = randomWalkDegreeLabels(degreeGraph, int(walkLength * (1 - alpha)), random_walk_count)
        walk = randomWalkDegreeLabels(subgraph, int(alpha * walkLength), random_walk_count)
        walkFile.write(arr2str(walk + degreeWalk) + "\n")
        indexToName[index] = graph.graph['name']
    print()
    walkFile.close()

    return indexToName


def saveVectors(vectors, outputfile, IdToName):
    output = open(outputfile, 'w')

    output.write(str(len(vectors)) + "\n")
    for i in range(len(vectors)):
        output.write(str(IdToName[i]))
        for j in vectors[i]:
            output.write('\t' + str(j))
        output.write('\n')
    output.close()


def structural_embedding(args):
    inputDir = args.input
    outputFile = args.output
    iterations = args.iter
    dimensions = args.d
    window = args.windowSize
    dm = 1 if args.model == 'dm' else 0
    indexToName = generateWalkFile(inputDir, args.walkLength, args.p)
    sentences = doc.TaggedLineDocument(inputDir + '.walk')

    model = doc.Doc2Vec(sentences, size=dimensions, iter=iterations, dm=dm, window=window)

    saveVectors(list(model.docvecs), outputFile, indexToName)
