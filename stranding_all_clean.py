# First networkx library is imported 
# along with matplotlib
import numpy as np
import networkx as nx
import netgraph as ng
import matplotlib.pyplot as plt
import copy
import pdb

# number of vertices
v = 2

# number of edges
edgeNum = 1

# vertices that are on the boundary
boundary = [0,1]

# stranding
w1 = [1,0,0]
w2 = [1,1,0]

# initializing adjacency matrix
adj = np.zeros((v,v))

# Test Cases

##test 0, v=2, b=0,1, eN=1
adj[0,1] = 1
adj[1,0] = -1

##test 1, v=4, b=1,2,3, eN=3
##adj[1,0] = 1
##adj[2,0] = 1
##adj[3,0] = 1
##adj[0,1] = -1
##adj[0,2] = -1
##adj[0,3] = -1

##test 2, v=4, b=1,2,3, eN=3
##adj[1,0] = -1
##adj[2,0] = -1
##adj[3,0] = -1
##adj[0,1] = 1
##adj[0,2] = 1
##adj[0,3] = 1

##test 3, v=10, b=1,2,3,4,5,6, eN=9
##adj[1,0] = 1
##adj[2,8] = 1
##adj[3,8] = 1
##adj[4,9] = 1
##adj[5,9] = 1
##adj[6,0] = 1
##adj[7,0] = 1
##adj[7,8] = 1
##adj[7,9] = 1
##adj[0,1] = -1
##adj[8,2] = -1
##adj[8,3] = -1
##adj[9,4] = -1
##adj[9,5] = -1
##adj[0,6] = -1
##adj[0,7] = -1
##adj[8,7] = -1
##adj[9,7] = -1

####test 4, v=8, backtracking, b=1,2,3,4, eN=8
##adj[1,0] = 1
##adj[0,1] = -1
##
##adj[2,5] = -1
##adj[5,2] = 1
##
##adj[3,7] = 1
##adj[7,3] = -1
##
##adj[4,6] = -1
##adj[6,4] = 1
##
##adj[0,5] = -1
##adj[5,0] = 1
##
##adj[0,6] = -1
##adj[6,0] = 1
##
##adj[6,7] = 1
##adj[7,6] = -1
##
##adj[7,5] = -1
##adj[5,7] = 1

##test 5, v=10, backtracking, b=1,2,3,4, eN=11
##adj[1,0] = 1
##adj[0,1] = -1
##
##adj[2,7] = -1
##adj[7,2] = 1
##
##adj[3,9] = -1
##adj[9,3] = 1
##
##adj[4,6] = 1
##adj[6,4] = -1
##
##adj[0,5] = -1
##adj[5,0] = 1
##
##adj[0,7] = -1
##adj[7,0] = 1
##
##adj[5,6] = 1
##adj[6,5] = -1
##
##adj[5,8] = 1
##adj[8,5] = -1
##
##adj[7,8] = 1
##adj[8,7] = -1
##
##adj[8,9] = -1
##adj[9,8] = 1
##
##adj[6,9] = -1
##adj[9,6] = 1

##test 6, v=16, b=1,2,3,4,5,6 eN=18
##adj[1,10] = 1
##adj[10,1] = -1
##
##adj[6,11] = -1
##adj[11,6] = 1
##
##adj[2,0] = 1
##adj[0,2] = -1
##
##adj[3,8] = -1
##adj[8,3] = 1
##
##adj[4,14] = 1
##adj[14,4] = -1
##
##adj[5,15] = -1
##adj[15,5] = 1
##
##adj[0,8] = -1
##adj[8,0] = 1
##
##adj[0,7] = -1
##adj[7,0] = 1
##
##adj[8,9] = 1
##adj[9,8] = -1
##
##adj[7,9] = 1
##adj[9,7] = -1
##
##adj[7,10] = 1
##adj[10,7] = -1
##
##adj[9,13] = -1
##adj[13,9] = 1
##
##adj[10,11] = -1
##adj[11,10] = 1
##
##adj[11,12] = 1
##adj[12,11] = -1
##
##adj[12,13] = -1
##adj[13,12] = 1
##
##adj[12,15] = -1
##adj[15,12] = 1
##
##adj[15,14] = 1
##adj[14,15] = -1
##
##adj[13,14] = 1
##adj[14,13] = -1

###test 7, v=12, b=1,2,3,4,5,6 eN=12
##adj[1,0] = 1
##adj[0,1] = -1
##
##adj[6,7] = -1
##adj[7,6] = 1
##
##adj[2,11] = -1
##adj[11,2] = 1
##
##adj[3,10] = 1
##adj[10,3] = -1
##
##adj[4,9] = -1
##adj[9,4] = 1
##
##adj[5,8] = 1
##adj[8,5] = -1
##
##adj[0,7] = -1
##adj[7,0] = 1
##
##adj[0,11] = -1
##adj[11,0] = 1
##
##adj[11,10] = 1
##adj[10,11] = -1
##
##adj[10,9] = -1
##adj[9,10] = 1
##
##adj[9,8] = 1
##adj[8,9] = -1
##
##adj[8,7] = -1
##adj[7,8] = 1

### test 8, v=18, b=1,2,3,4,5,6,7,8,9,10, eN=17
##adj[1,0] = 1
##adj[0,1] = -1
##
##adj[2,14] = -1
##adj[14,2] = 1
##
##adj[3,14] = -1
##adj[14,3] = 1
##
##adj[4,15] = -1
##adj[15,4] = 1
##
##adj[5,15] = -1
##adj[15,5] = 1
##
##adj[6,16] = -1
##adj[16,6] = 1
##
##adj[7,16] = -1
##adj[16,7] = 1
##
##adj[8,17] = -1
##adj[17,8] = 1
##
##adj[9,17] = -1
##adj[17,9] = 1
##
##adj[10,0] = 1
##adj[0,10] = -1
##
##adj[0,11] = -1
##adj[11,0] = 1
##
##adj[11,12] = 1
##adj[12,11] = -1
##
##adj[11,13] = 1
##adj[13,11] = -1
##
##adj[12,14] = -1
##adj[14,12] = 1
##
##adj[12,15] = -1
##adj[15,12] = 1
##
##adj[13,16] = -1
##adj[16,13] = 1
##
##adj[13,17] = -1
##adj[17,13] = 1

### test 9, v=14, b=1,2,3,4,5,6,7,8, eN=13
##adj[1,0] = 1
##adj[0,1] = -1
##
##adj[2,11] = -1
##adj[11,2] = 1
##
##adj[3,11] = -1
##adj[11,3] = 1
##
##adj[4,12] = 1
##adj[12,4] = -1
##
##adj[5,12] = 1
##adj[12,5] = -1
##
##adj[6,13] = -1
##adj[13,6] = 1
##
##adj[7,13] = -1
##adj[13,7] = 1
##
##adj[8,9] = 1
##adj[9,8] = -1
##
##adj[10,0] = 1
##adj[0,10] = -1
##
##adj[10,9] = 1
##adj[9,10] = -1
##
##adj[11,0] = 1
##adj[0,11] = -1
##
##adj[12,10] = -1
##adj[10,12] = 1
##
##adj[13,9] = 1
##adj[9,13] = -1

### test 10, v=14, b=1,2,3,4,5,6,7,8, eN=13
##adj[1,0] = 1
##adj[0,1] = -1
##
##adj[2,10] = 1
##adj[10,2] = -1
##
##adj[3,12] = -1
##adj[12,3] = 1
##
##adj[4,12] = -1
##adj[12,4] = 1
##
##adj[5,13] = -1
##adj[13,5] = 1
##
##adj[6,13] = -1
##adj[13,6] = 1
##
##adj[7,11] = 1
##adj[11,7] = -1
##
##adj[8,0] = 1
##adj[0,8] = -1
##
##adj[9,0] = 1
##adj[0,9] = -1
##
##adj[10,9] = -1
##adj[9,10] = 1
##
##adj[11,9] = -1
##adj[9,11] = 1
##
##adj[12,10] = 1
##adj[10,12] = -1
##
##adj[13,11] = 1
##adj[11,13] = -1

### test 11, v=18, b=1,2,3,4,5,6,7,8, eN=19
##adj[1,0] = 1
##adj[0,1] = -1
##
##adj[2,10] = -1
##adj[10,2] = 1
##
##adj[3,12] = -1
##adj[12,3] = 1
##
##adj[4,12] = -1
##adj[12,4] = 1
##
##adj[5,15] = -1
##adj[15,5] = 1
##
##adj[6,17] = -1
##adj[17,6] = 1
##
##adj[7,17] = -1
##adj[17,7] = 1
##
##adj[8,14] = -1
##adj[14,8] = 1
##
##adj[0,9] = -1
##adj[9,0] = 1
##
##adj[0,10] = -1
##adj[10,0] = 1
##
##adj[10,11] = 1
##adj[11,10] = -1
##
##adj[9,11] = 1
##adj[11,9] = -1
##
##adj[9,13] = 1
##adj[13,9] = -1
##
##adj[11,12] = -1
##adj[12,11] = 1
##
##adj[13,14] = -1
##adj[14,13] = 1
##
##adj[13,15] = -1
##adj[15,13] = 1
##
##adj[15,16] = 1
##adj[16,15] = -1
##
##adj[14,16] = 1
##adj[16,14] = -1
##
##adj[16,17] = -1
##adj[17,16] = 1

### test 12, v=16, b=1,2,3,4, eN=20
##adj[1,0] = 1
##adj[0,1] = -1
##
##adj[2,5] = -1
##adj[5,2] = 1
##
##adj[3,8] = 1
##adj[8,3] = -1
##
##adj[4,9] = -1
##adj[9,4] = 1
##
##adj[0,5] = -1
##adj[5,0] = 1
##
##adj[0,11] = -1
##adj[11,0] = 1
##
##adj[5,6] = 1
##adj[6,5] = -1
##
##adj[6,15] = -1
##adj[15,6] = 1
##
##adj[6,7] = -1
##adj[7,6] = 1
##
##adj[7,8] = 1
##adj[8,7] = -1
##
##adj[7,14] = 1
##adj[14,7] = -1
##
##adj[8,9] = -1
##adj[9,8] = 1
##
##adj[9,10] = 1
##adj[10,9] = -1
##
##adj[10,11] = -1
##adj[11,10] = 1
##
##adj[10,13] = -1
##adj[13,10] = 1
##
##adj[11,12] = 1
##adj[12,11] = -1
##
##adj[12,15] = -1
##adj[15,12] = 1
##
##adj[12,13] = -1
##adj[13,12] = 1
##
##adj[13,14] = 1
##adj[14,13] = -1
##
##adj[14,15] = -1
##adj[15,14] = 1

def stranding():
    vertex = boundary[0]

    # create hash map for each possibility
    hashMap = {}
    for i in range(v):
        for j in range(i+1, v):
            if i == j:
                continue
            else:
                hashMap[(i,j)] = {}
    
    result = findFollowStrand(vertex, [], [], [], [], hashMap, [])

    valid_result = []

    has_loop = has_internal_loop(adj, boundary)

    if has_loop:
        for r in result:
            done = r[2]
            if len(done) == edgeNum:
                valid_result.append(r)
            else:
                Edges, Red_edge, Done, DoneB, HashMap = internalLoopsNew([r[0]], [r[1]], [r[2]], [r[3]], [r[4]])
                for k in range(len(Edges)):
                    if len(Done[k]) == edgeNum:
                        valid_result.append([Edges[k], Red_edge[k], Done[k], DoneB[k], HashMap[k]])
    else:
        for r in result:
            done = r[2]
            if len(done) == edgeNum:
                valid_result.append(r)
                #print(r[0])

    def remove_duplicates(valid_result):
        # Create a set to keep track of unique first items
        unique_first_items = set()

        # Initialize a new list to store the result
        duplicates = []

        # Iterate through the input list
        for inner_list in valid_result:
            # Get the first item of the inner list
            first_item = inner_list[0]
            
            # Convert the first item to a tuple of sorted tuples to make it hashable
            sorted_first_item = tuple(sorted(first_item))
            
            # Check if it's a new unique first item
            if sorted_first_item not in unique_first_items:
                unique_first_items.add(sorted_first_item)  # Add to the set of unique first items
                duplicates.append(inner_list)  # Add the entire inner list to the result
        return duplicates

    duplicates = remove_duplicates(valid_result)

    # Get binary word
    for d in duplicates:
        hashMap = d[4]
        binary_word = ""
        for b in boundary: # loop through boundary
            for j in range(len(adj[b])): # look at adjecency matrix for vertex j connected to boundary
                val = adj[b,j]
                if val != 0:
                    lst = []
                    tup = tuple(sorted((b,j)))
                    if 1 in hashMap[tup] and 2 in hashMap[tup]:
                        lst = [x - y for x, y in zip(w2, w1)]
                    elif 1 in hashMap[tup]:
                        lst = w1
                    else:
                        lst = [-1 * x + 1 for x in w2]

                    if val < 0:
                        lst = [-1 * x + 1 for x in lst]


                    indices_of_ones = [index for index, value in enumerate(lst) if value == 1]
                    indices_of_ones = [x + 1 for x in indices_of_ones]
                    binary_word += ''.join(map(str, indices_of_ones))
                    break
            else:
                continue
        d.append(int(binary_word))

    final_sorted = sorted(duplicates, key=lambda x: x[-1])

    # Determine if clockwise based on boundary vertices
    for f in final_sorted:
        i_found = (False, None)
        j_found = (False, None)
        clockwise = True
        f_edges = f[0]
        for tup in f_edges:
            i = tup[0]
            j = tup[1]

            if i in boundary:
                i_found = (True, i)

            if j in boundary:
                j_found = (True, j)

            if i_found[0] and j_found[0]:
                if j_found[1] < i_found[1]:
                    clockwise = False
                    break
                i_found = (False, None)
                j_found = (False, None)
            
        f.append(clockwise)
                

    # Print the result
    
    for unique_list in final_sorted:
        print(unique_list[0])
        print(unique_list[1])
        print(unique_list[5])
        print(unique_list[6])
        print("\n")

# code to print out all generated strandings
##    print(final_sorted[0][0])
##    print(final_sorted[0][1])
##    print(final_sorted[0][5])
##    print(final_sorted[0][6])
##    print("\n")
##    least = final_sorted[0][5]
##    going = True
##    n = 1
##
##    while going:
##        if final_sorted[n][5] == least:
##            print(final_sorted[n][0])
##            print(final_sorted[n][1])
##            print(final_sorted[n][5])
##            print(final_sorted[n][6])
##            print("\n")
##
##            n += 1
##        else:
##            going = False

    print(len(final_sorted))
    return final_sorted

def getStrand(vertex, edges, red_edge, done, doneB, hashMap):
    for j in range(len(adj[0])): # loop through neighbors of vertex
        edge = adj[vertex, j] # look at edge weight per neighbor
        tup = tuple(sorted((vertex, j)))

        # need to add creating copy of hashmap with strands added
        if edge == 1 or edge == -2:
            e1 = edges.copy()
            e1.append((vertex, j))
            re1 = red_edge.copy()
            d1 = done.copy()
            d1.append((vertex, j))
            dB1 = doneB.copy()
            dB1.append(vertex)

            e2 = edges.copy()
            e2.append((j, vertex))
            re2 = red_edge.copy()
            re2.append((j, vertex))
            d2 = done.copy()
            d2.append((j, vertex))
            dB2 = doneB.copy()
            dB2.append(vertex)

            e3 = edges.copy()
            e3.append((j, vertex))
            re3 = red_edge.copy()
            d3 = done.copy()
            dB3 = doneB.copy()
            
            hM1 = copy.deepcopy(hashMap)
            hM2 = copy.deepcopy(hashMap)
            hM3 = copy.deepcopy(hashMap)

            hM1[tup][1] = (vertex, j)
            hM2[tup][2] = (j, vertex)
            hM3[tup][1] = (j, vertex)
            
            # [edges, red_edge, done, doneB, hashMap, queue j, strand, inStrand, red]
            w1 = [e1, re1, d1, dB1, hM1, [], j, ((vertex, j), 1), True, False] # in w1 strand
            w2 = [e2, re2, d2, dB2, hM2, [], j, ((j, vertex), 2), False, True] # out w2 strand
            w1w2 = [e3, re3, d3, dB3, hM3, [[((vertex, j), 2), True, j, vertex]], j, ((j, vertex), 1), False, False] # in w2 strand and out w1 strand
            return [w1, w2, w1w2]
            
        elif edge == 2 or edge == -1:
            e1 = edges.copy()
            e1.append((j, vertex))
            re1 = red_edge.copy()
            d1 = done.copy()
            d1.append((j, vertex))
            dB1 = doneB.copy()
            dB1.append(vertex)

            e2 = edges.copy()
            e2.append((vertex, j))
            re2 = red_edge.copy()
            re2.append((vertex, j))
            d2 = done.copy()
            d2.append((vertex, j))
            dB2 = doneB.copy()
            dB2.append(vertex)

            e3 = edges.copy()
            e3.append((vertex, j))
            re3 = red_edge.copy()
            d3 = done.copy()
            dB3 = doneB.copy()
            
            hM1 = copy.deepcopy(hashMap)
            hM2 = copy.deepcopy(hashMap)
            hM3 = copy.deepcopy(hashMap)

            hM1[tup][1] = (j, vertex)
            hM2[tup][2] = (vertex, j)
            hM3[tup][1] = (vertex, j)
            
            # [edges, red_edge, done, doneB, hashMap, j, strand, inStrand, red]
            w1 = [e1, re1, d1, dB1, hM1, [], j, ((j, vertex), 1), False, False] # out w1 strand
            w2 = [e2, re2, d2, dB2, hM2, [], j, ((vertex, j), 2), True, True] # in w2 strand
            w1w2 = [e3, re3, d3, dB3, hM3, [[((j, vertex), 2), False, j, vertex]], j, ((vertex, j), 1), True, False] # in w1 strand, out w2 strand
            return [w1, w2, w1w2]

def findFollowStrand(vertex, e, re, d, dB, hM, result):
    lst = getStrand(vertex, e, re, d, dB, hM)
    
    for i in range(len(lst)):
        possibility = lst[i]
        
        edges = possibility[0]
        red_edge = possibility[1]
        done = possibility[2]
        doneB = possibility[3]
        hashMap = possibility[4]
        queue = possibility[5]
        j = possibility[6]
        strand = possibility[7]
        inStrand = possibility[8]
        red = possibility[9]

        # check if need to create a new strand
        if j in boundary:
            doneB.append(j)

            while len(queue) != 0:
                item = queue[0]
                new_strand = item[0]
                in_new = item[1]
                newV = item[2]
                origin = item[3]
                newTup = tuple(sorted((newV, origin)))

                queue = queue[1:]
                
                if new_strand[1] not in hashMap[newTup] and 3 - new_strand[1] in hashMap[newTup]: # other strand in queue has not been added
                    edges.append(new_strand[0])
                    red_edge.append(new_strand[0])
                    done.append(new_strand[0])
                    doneB.append(origin)
                    hashMap[newTup][new_strand[1]] = new_strand[0]
                
            if len(doneB) == len(boundary):
                result.append([edges, red_edge, done, doneB, hashMap])
            else:
                for b in boundary: # check boundary vertices
                    if b not in doneB:
                        result2 = findFollowStrand(b, edges, red_edge, done, doneB, hashMap, [])
                        for i in range(len(result2)):
                            Edges2 = result[i][0]
                            Red_edge2 = result[i][1]
                            Done2 = result[i][2]
                            DoneB2 = result[i][3]
                            HashMap2 = result[i][4]
                            
                            result.append([Edges2, Red_edge2, Done2, DoneB2, HashMap2])
            continue
        else:
            undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(False, None, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, j, strand, inStrand, red) # make these a list until hashMap
            for k in range(len(Edges2)):
                result.append([Edges2[k], Red_edge2[k], Done2[k], DoneB2[k], HashMap2[k]])
            continue
    return result

def checkEdge(done, hashMap, vertex, j, strand, inStrand, tup): # add some more valid checks such as with direction of edge
    other = 3 - strand[1]
        
    if (j,vertex) not in done and (vertex,j) not in done and strand[1] not in hashMap[tup]:
        # checks if other strand is in edge and prevents having two strands go in the same direction
        if other in hashMap[tup]:
            if inStrand:
                if (vertex, j) == hashMap[tup][other]: 
                    return False
                else:
                    return True
            else:
                if (j, vertex) == hashMap[tup][other]:
                    return False
                else:
                    return True
        else:
            return True
    else:
        return False

def followStrand(internal, iStart, undo, Edges, Red_edge, Done, DoneB, HashMap, Queue, vertex, strand, inStrand, red = False):
    if len(Done[0]) == edgeNum: # checks if number of stranded edges is equal to number of total strands in web
        return False, Edges, Red_edge, Done, DoneB, HashMap # done stranding everything

##    print(Edges[0])
##    print(DoneB[0])
##    print(Queue)
    
    for j in range(len(adj[0])): # loop through neighbors of vertex
        edge = adj[vertex, j] # look at edge weight per neighbor
        tup = tuple(sorted((vertex, j)))

        if edge != 0 and checkEdge(Done[0], HashMap[0], vertex, j, strand, inStrand, tup):
            edges = Edges[0].copy()
            red_edge = Red_edge[0].copy()
            done = Done[0].copy()
            doneB = DoneB[0].copy()
            hashMap = copy.deepcopy(HashMap[0])
            queue = Queue.copy()
            
            if inStrand:
                hashMap[tup][strand[1]] = (vertex, j)
                edges.append((vertex, j))
                if red:
                     red_edge.append((vertex, j))
                if j in boundary: # check if boundary vertex, meaning strand is done
                    doneB.append(j)
                    done.append((vertex, j))
                    
                    # meaning w1 and edge going opposite
                    if edge < 0 and strand[1] == 1 and 2 not in hashMap[tup]:
                        edges.append((j, vertex))
                        red_edge.append((j, vertex))
                        hashMap[tup][2] = (j, vertex)
                        
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, vertex, ((j, vertex), 2), True, True) # follow w2 strand going out
                        if not undo:
                            Edges = Edges + Edges2
                            Red_edge = Red_edge + Red_edge2
                            Done = Done + Done2
                            DoneB = DoneB + DoneB2
                            HashMap = HashMap + HashMap2
                        continue
                    # meaning w2 and edge going same
                    elif edge > 0 and strand[1] == 2 and 1 not in hashMap[tup]:
                        edges.append((j, vertex))
                        hashMap[tup][1] = (j, vertex)

                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, vertex, ((j, vertex), 1), True) # follow w1 strand going in

                        if not undo:
                            Edges = Edges + Edges2
                            Red_edge = Red_edge + Red_edge2
                            Done = Done + Done2
                            DoneB = DoneB + DoneB2
                            HashMap = HashMap + HashMap2
                        continue
                    else: #recurse on another boundary vertex
                        count = 0
                        while len(queue) != 0 and count == 0:
                            item = queue[0]
                            new_strand = item[0]
                            in_new = item[1]
                            newV = item[2]
                            origin = item[3]
                            newTup = tuple(sorted((newV, origin)))

                            queue = queue[1:]
                            
                            if new_strand[1] not in hashMap[newTup] and 3 - new_strand[1] in hashMap[newTup]: # other strand in queue has not been added
                                edges.append(new_strand[0])
                                red_edge.append(new_strand[0])
                                done.append(new_strand[0])
                                doneB.append(origin)
                                hashMap[newTup][new_strand[1]] = new_strand[0]

                                new_red = False

                                if new_strand[1] == 2:
                                    new_red = True

                                undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, newV, new_strand, in_new, new_red)

                                Edges = Edges + Edges2
                                Red_edge = Red_edge + Red_edge2
                                Done = Done + Done2
                                DoneB = DoneB + DoneB2
                                HashMap = HashMap + HashMap2

                                count += 1
                            
                        if count == 0: # other strand in queue has been added
                            if len(doneB) == len(boundary):
                                Edges.append(edges)
                                Red_edge.append(red_edge)
                                Done.append(done)
                                DoneB.append(doneB)
                                HashMap.append(hashMap)
                            else:
                                for b in boundary: # check boundary vertices
                                    if b not in doneB:
                                        result = findFollowStrand(b, edges, red_edge, done, doneB, hashMap, [])
                                        for i in range(len(result)):
                                            Edges2 = result[i][0]
                                            Red_edge2 = result[i][1]
                                            Done2 = result[i][2]
                                            DoneB2 = result[i][3]
                                            HashMap2 = result[i][4]
                                            
                                            Edges.append(Edges2)
                                            Red_edge.append(Red_edge2)
                                            Done.append(Done2)
                                            DoneB.append(DoneB2)
                                            HashMap.append(HashMap2)

                        continue
        
                else: # strand not done, j not in boundary
                    # meaning w1 and edge going opposite but has w2, or w2 and edge going same but has w1, or w1 and edge going same direction, or w2 and edge going opposite
                    if (edge > 0 and strand[1] == 2 and 1 in hashMap[tup]) \
                       or (edge < 0 and strand[1] == 1 and 2 in hashMap[tup]) \
                       or (edge > 0 and strand[1] == 1) or (edge < 0 and strand[1] == 2):
                        done.append((vertex, j))
                    #go start on j
                    if red:
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, j, strand, True, True)

                        if not undo:
                            Edges = Edges + Edges2
                            Red_edge = Red_edge + Red_edge2
                            Done = Done + Done2
                            DoneB = DoneB + DoneB2
                            HashMap = HashMap + HashMap2
                        continue
                    else:
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, j, strand, True)

                        if not undo:
                            Edges = Edges + Edges2
                            Red_edge = Red_edge + Red_edge2
                            Done = Done + Done2
                            DoneB = DoneB + DoneB2
                            HashMap = HashMap + HashMap2
                        continue
            else: # going out direction
                hashMap[tup][strand[1]] = (j, vertex)
                edges.append((j, vertex))
                if red:
                     red_edge.append((j, vertex))
                if j in boundary: # check if boundary vertex, meaning strand is done
                    doneB.append(j)
                    done.append((j, vertex))

                    if edge > 0 and strand[1] == 1 and 2 not in hashMap[tup]: # meaning w1 and edge going opposite
                        edges.append((vertex, j))
                        red_edge.append((vertex, j))
                        hashMap[tup][2] = (vertex, j)
                        
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, vertex, ((vertex, j), 2), False, True) # follow w2 strand going in

                        if not undo:
                            Edges = Edges + Edges2
                            Red_edge = Red_edge + Red_edge2
                            Done = Done + Done2
                            DoneB = DoneB + DoneB2
                            HashMap = HashMap + HashMap2
                        continue
                    # meaning w2 and edge going same
                    elif edge < 0 and strand[1] == 2 and 1 not in hashMap[tup]:
                        edges.append((vertex, j))
                        hashMap[tup][1] = (vertex, j)
                        
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, vertex, ((vertex, j), 1), False) # follow w1 strand going out

                        if not undo:
                            Edges = Edges + Edges2
                            Red_edge = Red_edge + Red_edge2
                            Done = Done + Done2
                            DoneB = DoneB + DoneB2
                            HashMap = HashMap + HashMap2
                        continue
                    else: #recurse on another boundary vertex or follow next strand in queue
                        count = 0
                        while len(queue) != 0 and count == 0:
                            item = queue[0]
                            new_strand = item[0]
                            in_new = item[1]
                            newV = item[2]
                            origin = item[3]
                            newTup = tuple(sorted((newV, origin)))

                            queue = queue[1:]
                            
                            if new_strand[1] not in hashMap[newTup] and 3 - new_strand[1] in hashMap[newTup]: # other strand in queue has not been added
                                edges.append(new_strand[0])
                                red_edge.append(new_strand[0])
                                done.append(new_strand[0])
                                doneB.append(origin)
                                hashMap[newTup][new_strand[1]] = new_strand[0]

                                new_red = False

                                if new_strand[1] == 2:
                                    new_red = True

                                undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, newV, new_strand, in_new, new_red)

                                Edges = Edges + Edges2
                                Red_edge = Red_edge + Red_edge2
                                Done = Done + Done2
                                DoneB = DoneB + DoneB2
                                HashMap = HashMap + HashMap2

                                count += 1
                            
                        if count == 0: # other strand in queue has been added
                            if len(doneB) == len(boundary):
                                Edges.append(edges)
                                Red_edge.append(red_edge)
                                Done.append(done)
                                DoneB.append(doneB)
                                HashMap.append(hashMap)
                            else:
                                for b in boundary: # check boundary vertices
                                    if b not in doneB:
                                        result = findFollowStrand(b, edges, red_edge, done, doneB, hashMap, [])
                                        for i in range(len(result)):
                                            Edges2 = result[i][0]
                                            Red_edge2 = result[i][1]
                                            Done2 = result[i][2]
                                            DoneB2 = result[i][3]
                                            HashMap2 = result[i][4]

                                            Edges.append(Edges2)
                                            Red_edge.append(Red_edge2)
                                            Done.append(Done2)
                                            DoneB.append(DoneB2)
                                            HashMap.append(HashMap2)

                        continue
                else:
                    # meaning w1 and edge going opposite but has w2, or w2 and edge going same but has w1, or w1 and edge going same, or w2 and edge going opposite
                    if (edge < 0 and strand[1] == 2 and 1 in hashMap[tup]) \
                       or (edge > 0 and strand[1] == 1 and 2 in hashMap[tup]) \
                       or (edge < 0 and strand[1] == 1) \
                       or (edge > 0 and strand[1] == 2):
                        done.append((j, vertex))
                        
                    #go start on j
                    if red:
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, j, strand, False, True)

                        if not undo:
                            Edges = Edges + Edges2
                            Red_edge = Red_edge + Red_edge2
                            Done = Done + Done2
                            DoneB = DoneB + DoneB2
                            HashMap = HashMap + HashMap2
                        continue
                    else:
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(internal, iStart, False, [edges], [red_edge], [done], [doneB], [hashMap], queue, j, strand, False)

                        if not undo:
                            Edges = Edges + Edges2
                            Red_edge = Red_edge + Red_edge2
                            Done = Done + Done2
                            DoneB = DoneB + DoneB2
                            HashMap = HashMap + HashMap2
                        continue
    if len(Edges) == 1:
        if internal:
            if vertex == iStart:
                return False, Edges, Red_edge, Done, DoneB, HashMap
            else:
                return True, Edges, Red_edge, Done, DoneB, HashMap
        else:
            return True, Edges[1:], Red_edge[1:], Done[1:], DoneB[1:], HashMap[1:] # have to backtrack, no available vertex to continue strand
    else:
        return False, Edges[1:], Red_edge[1:], Done[1:], DoneB[1:], HashMap[1:]


def internalLoopsNew(Edges, Red_edge, Done, DoneB, HashMap):
    edges = Edges[0].copy()
    red_edge = Red_edge[0].copy()
    done = Done[0].copy()
    doneB = DoneB[0].copy()
    hashMap = copy.deepcopy(HashMap[0])

    for edge in Edges[0]:
        i = edge[0]
        j = edge[1]

        if i not in boundary and j not in boundary:
            if (i,j) not in Done[0] and (j,i) not in Done[0]:
                edgeValue = adj[i,j]
                tup = tuple(sorted((i,j)))
                if edge in red_edge: # means it has a w2 strand in that edge
                    # create w1 in opposite direction
                    if edgeValue < 0:
                        edges.append((i, j))
                        done.append((i, j))
                        strand = ((i, j), 1)
                        hashMap[tup][1] = (i,j)
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(True, i, False, [edges], [red_edge], [done], [doneB], [hashMap], [], j, strand, True, False)
                    else:
                        edges.append((j, i))
                        done.append((j, i))
                        strand = ((j, i), 1)
                        hashMap[tup][1] = (j,i)
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(True, j, False, [edges], [red_edge], [done], [doneB], [hashMap], [], i, strand, True, False)

                    for k in range(len(Edges2)):
                        Edges3, Red_edge3, Done3, DoneB3, HashMap3 = internalLoopsNew([Edges2[k]], [Red_edge2[k]], [Done2[k]], [DoneB2[k]], [HashMap2[k]])
                        Edges = Edges + Edges3
                        Red_edge = Red_edge + Red_edge3
                        Done = Done + Done3
                        DoneB = DoneB + DoneB3
                        HashMap = HashMap + HashMap3
                    return Edges, Red_edge, Done, DoneB, HashMap
                                            
                else: # means w1 is in that edge
                    # create a w2 strand in same direction as edge
                    if edgeValue < 0:
                        edges.append((j, i))
                        red_edge.append((j, i))
                        done.append((j, i))
                        strand = ((j, i), 2)
                        hashMap[tup][2] = (j,i)
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(True, j, False, [edges], [red_edge], [done], [doneB], [hashMap], [], i, strand, True, True)
                    else:
                        edges.append((i, j))
                        red_edge.append((i, j))
                        done.append((i, j))
                        strand = ((i, j), 2)
                        hashMap[tup][2] = (i,j)
                        undo, Edges2, Red_edge2, Done2, DoneB2, HashMap2 = followStrand(True, i, False, [edges], [red_edge], [done], [doneB], [hashMap], [], j, strand, True, True)

                    for k in range(len(Edges2)):
                        Edges3, Red_edge3, Done3, DoneB3, HashMap3 = internalLoopsNew([Edges2[k]], [Red_edge2[k]], [Done2[k]], [DoneB2[k]], [HashMap2[k]])
                        Edges = Edges + Edges3
                        Red_edge = Red_edge + Red_edge3
                        Done = Done + Done3
                        DoneB = DoneB + DoneB3
                        HashMap = HashMap + HashMap3
                    return Edges, Red_edge, Done, DoneB, HashMap
                
    return Edges, Red_edge, Done, DoneB, HashMap

# checks if there is a loop in the web using just the internal vertices
def has_internal_loop(adj_matrix, boundary_vertices):
    def dfs(node, visited, parent):
        visited[node] = True
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[node, neighbor] != 0:  # Check for a nonzero value in the adjacency matrix
                if not visited[neighbor]:
                    if neighbor in boundary_vertices:
                        continue  # Skip boundary vertices
                    if dfs(neighbor, visited, node):
                        return True
                elif neighbor != parent:
                    return True
        return False

    n = len(adj_matrix)
    visited = [False] * n

    for node in range(n):
        if not visited[node] and node not in boundary_vertices:
            if dfs(node, visited, -1):
                return True

    return False

stranding()

## code to plot stranding result
##res = stranding()
##n=5
##
##edges = res[n][0]
##red_edge = res[n][1]
##
##G = nx.DiGraph(edges)
##
##edge_color = dict()
##for i, edge in enumerate(G.edges):
##    edge_color[edge] = 'tab:red' if edge in red_edge else 'tab:blue'
##
##node_label = dict()
##for n in G.nodes:
##    node_label[n] = n
##
##plt_instance = ng.InteractiveGraph(G, edge_color=edge_color, node_labels=node_label,
##                                   arrows=True)
##plt.show()
