#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
import copy
import math

import chess
import board
import numpy as np
import sys
import queue
from typing import List
from collections import defaultdict
from tqdm import trange
import pandas as pd

RawStateType = List[List[List[int]]]

from itertools import permutations


class Aichess():
    """
    A class to represent the game of chess.

    ...

    Attributes:
    -----------
    chess : Chess
        represents the chess game
        
    listNextStates : list
        List of next possible states for the current player.

    listVisitedStates : list
        List of all visited states during A*.

    listVisitedSituations : list
        List of visited game situations (state + color) for minimax/alpha-beta pruning.

    pathToTarget : list
        Sequence of states from the initial state to the target (used by A*).

    depthMax : int
        Maximum search depth for minimax/alpha-beta searches.

    dictPath : dict
        Dictionary used to reconstruct the path in A* search.

    Methods:
    --------
    copyState(state) -> list
        Returns a deep copy of the given state.

    isVisitedSituation(color, mystate) -> bool
        Checks whether a given state with a specific color has already been visited.

    getListNextStatesW(myState) -> list
        Returns a list of possible next states for the white pieces.

    getListNextStatesB(myState) -> list
        Returns a list of possible next states for the black pieces.

    isSameState(a, b) -> bool
        Checks whether two states represent the same board configuration.

    isVisited(mystate) -> bool
        Checks if a given state has been visited in search algorithms.

    getCurrentState() -> list
        Returns the combined state of both white and black pieces.

    isCheckMate(mystate) -> bool
        Determines if a state represents a checkmate configuration.

    heuristica(currentState, color) -> int
        Calculates a heuristic value for the current state from the perspective of the given color.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces along the path between two states.

    changeState(start, to) -> None
        Moves a single piece from start state to to state.

    reconstructPath(state, depth) -> None
        Reconstructs the path from initial state to the target state for A*.

    h(state) -> int       
        Heuristic function for A* search.

    DepthFirstSearch(currentState, depth) -> bool
        Depth-first search algorithm.

    worthExploring(state, depth) -> bool
        Checks if a state is worth exploring during search using the optimised DFS algorithm.

    DepthFirstSearchOptimized(currentState, depth) -> bool
        Optimized depth-first search algorithm.

    BreadthFirstSearch(currentState, depth) -> None
        Breadth-first search algorithm.

    AStarSearch(currentState) 
        A* search algorithm -> To be implemented by you

    """

    def __init__(self, TA, myinit=True):

        if myinit:
            self.chess = chess.Chess(TA, True)
        else:
            self.chess = chess.Chess([], False)

        self.listNextStates = []
        self.listVisitedStates = []
        self.listVisitedSituations = []
        self.pathToTarget = []
        self.depthMax = 8;
        # Dictionary to reconstruct the visited path
        self.dictPath = {}
        # Prepare a dictionary to control the visited state and at which
        # depth they were found for DepthFirstSearchOptimized
        self.dictVisitedStates = {}
        self.qTable = defaultdict(lambda: defaultdict(float))
        

    def copyState(self, state):
        
        copyState = []
        for piece in state:
            copyState.append(piece.copy())
        return copyState
        
    def isVisitedSituation(self, color, mystate):
        
        if (len(self.listVisitedSituations) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedSituations)):
                    if self.isSameState(list(perm_state[j]), self.listVisitedSituations.__getitem__(k)[1]) and color == \
                            self.listVisitedSituations.__getitem__(k)[0]:
                        isVisited = True

            return isVisited
        else:
            return False

    def getListNextStatesW(self, myState):

        self.chess.boardSim.getListNextStatesW(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def getListNextStatesB(self, myState):
        self.chess.boardSim.getListNextStatesB(myState)
        self.listNextStates = self.chess.boardSim.listNextStates.copy()

        return self.listNextStates

    def isSameState(self, a, b):

        isSameState1 = True
        # a and b are lists
        for k in range(len(a)):

            if a[k] not in b:
                isSameState1 = False

        isSameState2 = True
        # a and b are lists
        for k in range(len(b)):

            if b[k] not in a:
                isSameState2 = False

        isSameState = isSameState1 and isSameState2
        return isSameState

    def isVisited(self, mystate):

        if (len(self.listVisitedStates) > 0):
            perm_state = list(permutations(mystate))

            isVisited = False
            for j in range(len(perm_state)):

                for k in range(len(self.listVisitedStates)):

                    if self.isSameState(list(perm_state[j]), self.listVisitedStates[k]):
                        isVisited = True

            return isVisited
        else:
            return False

    # Function for check mate for exercise 1 (white king is missing)
    def isCheckMate(self, mystate):
        
        # list of possible check mate states
        listCheckMateStates = [[[0,0,2],[2,4,6]],[[0,1,2],[2,4,6]],[[0,2,2],[2,4,6]],[[0,6,2],[2,4,6]],[[0,7,2],[2,4,6]]]

        # Check all state permuations and if they coincide with a list of CheckMates
        for permState in list(permutations(mystate)):
            if list(permState) in listCheckMateStates:
                return True

        return False   

    def newBoardSim(self, listStates):
        # We create a  new boardSim
        TA = np.zeros((8, 8))
        for state in listStates:
            TA[state[0]][state[1]] = state[2]

        self.chess.newBoardSim(TA)

    def getPieceState(self, state, piece):
        pieceState = None
        for i in state:
            if i[2] == piece:
                pieceState = i
                break
        return pieceState

    def getCurrentState(self):
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates
    
    def getCurrentStateSim(self):
        listStates = []
        for i in self.chess.boardSim.currentStateW:
            listStates.append(i)
        for j in self.chess.boardSim.currentStateB:
            listStates.append(j)
        return listStates

    def getNextPositions(self, state):
        # Given a state, we check the next possible states
        # From these, we return a list with position, i.e., [row, column]
        if state == None:
            return None
        if state[2] > 6:
            nextStates = self.getListNextStatesB([state])
        else:
            nextStates = self.getListNextStatesW([state])
        nextPositions = []
        for i in nextStates:
            nextPositions.append(i[0][0:2])
        return nextPositions

    def getWhiteState(self, currentState):
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState != None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState != None:
            blackState.append(brState)
        return blackState

    def getMovement(self, state, nextState):
        # Given a state and a successor state, return the postiion of the piece that has been moved in both states
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext != None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break

        return [pieceState, pieceNextState]

    def movePieces(self, start, depthStart, to, depthTo):
        
        # To move from one state to the next we will need to find
        # the state in common, and then move until the node 'to'
        moveList = []
        # We want that the depths are equal to find a common ancestor
        nodeTo = to
        nodeStart = start
        # if the depth of the node To is larger than that of start, 
        # we pick the ancesters of the node until being at the same
        # depth
        while(depthTo > depthStart):
            moveList.insert(0,to)
            nodeTo = self.dictPath[str(nodeTo)][0]
            depthTo-=1
        # Analogous to the previous case, but we trace back the ancestors
        #until the node 'start'
        while(depthStart > depthTo):
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # We move the piece the the parerent state of nodeStart
            self.changeState(nodeStart, ancestreStart)
            nodeStart = ancestreStart
            depthStart -= 1

        moveList.insert(0,nodeTo)
        # We seek for common node
        while nodeStart != nodeTo:
            ancestreStart = self.dictPath[str(nodeStart)][0]
            # Move the piece the the parerent state of nodeStart
            self.changeState(nodeStart,ancestreStart)
            # pick the parent of nodeTo
            nodeTo = self.dictPath[str(nodeTo)][0]
            # store in the list
            moveList.insert(0,nodeTo)
            nodeStart = ancestreStart
        # Move the pieces from the node in common
        # until the node 'to'
        for i in range(len(moveList)):
            if i < len(moveList) - 1:
                self.changeState(moveList[i],moveList[i+1])

    def reconstructPath(self, state, depth):
        # Once the solution is found, reconstruct the path taken to reach it
        for i in range(depth):
            self.pathToTarget.insert(0, state)
            # For each node, retrieve its parent from dictPath
            state = self.dictPath[str(state)][0]

        # Insert the root node at the beginning
        self.pathToTarget.insert(0, state)

    def h(self, state):

        if state[0][2] == 2:
            kingPosition = state[1]
            rookPosition = state[0]
        else:
            kingPosition = state[0]
            rookPosition = state[1]

        # Example heuristic assusiming the target position for the king is (2,4).

        # Calculate the Manhattan distance for the king to reach the target configuration (2,4)
        rowDiff = abs(kingPosition[0] - 2)
        colDiff = abs(kingPosition[1] - 4)
        # The minimum of row and column differences corresponds to diagonal moves,
        # and the absolute difference corresponds to remaining straight moves
        hKing = min(rowDiff, colDiff) + abs(rowDiff - colDiff)

        # Heuristic for the rook, with three different cases
        if rookPosition[0] == 0 and (rookPosition[1] < 3 or rookPosition[1] > 5):
            hRook = 0
        elif rookPosition[0] != 0 and 3 <= rookPosition[1] <= 5:
            hRook = 2
        else:
            hRook = 1

        # Total heuristic is the sum of king and rook heuristics
        return hKing + hRook

    def changeState(self, start, to):
        # Determine which piece has moved from the start state to the next state
        if start[0] == to[0]:
            movedPieceStart = 1
            movedPieceTo = 1
        elif start[0] == to[1]:
            movedPieceStart = 1
            movedPieceTo = 0
        elif start[1] == to[0]:
            movedPieceStart = 0
            movedPieceTo = 1
        else:
            movedPieceStart = 0
            movedPieceTo = 0

        # Move the piece that changed
        self.chess.moveSim(start[movedPieceStart], to[movedPieceTo])       

    def DepthFirstSearch(self, currentState, depth):
        # We visited the node, therefore we add it to the list
        # In DF, when we add a node to the list of visited, and when we have
        # visited all nodes, we remove it from the list of visited ones
        self.listVisitedStates.append(currentState)

        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        if depth + 1 <= self.depthMax:
            for son in self.getListNextStatesW(currentState):
                if not self.isVisited(son):
                    # in the state 'son', the first piece is the one just moved
                    # We check which piece in currentState matches the one moved
                    if son[0][2] == currentState[0][2]:
                        movedPieceIndex = 0
                    else:
                        movedPieceIndex = 1

                    # we move the piece to the new position
                    self.chess.moveSim(currentState[movedPieceIndex], son[0])
                    # We call the method again with 'son', increasing depth
                    if self.DepthFirstSearch(son, depth + 1):
                        # If the method returns True, this means that there has
                        # been a checkmate
                        # We add the state to the pathToTarget
                        self.pathToTarget.insert(0, currentState)
                        return True
                    # we reset the board to the previous state
                    self.chess.moveSim(son[0], currentState[movedPieceIndex])

        # We remove the node from the list of visited nodes
        # since we explored all successors
        self.listVisitedStates.remove(currentState)


    def worthExploring(self, state, depth):
        # First of all, check that the depth is not bigger than depthMax
        if depth > self.depthMax:
            return False
        visited = False
        # check if the state has been visited
        for perm in list(permutations(state)):
            permStr = str(perm)
            if permStr in list(self.dictVisitedStates.keys()):
                visited = True
                # If the state has been visited at a larger depth,
                # we are interested in visiting it again
                if depth < self.dictVisitedStates[perm]:
                    # Update the depth associated with the state
                    self.dictVisitedStates[permStr] = depth
                    return True
        # If never visited, add it to the dictionary at the current depth
        if not visited:
            permStr = str(state)
            self.dictVisitedStates[permStr] = depth
            return True


    def DepthFirstSearchOptimized(self, currentState, depth):
        # is it checkmate?
        if self.isCheckMate(currentState):
            self.pathToTarget.append(currentState)
            return True

        for son in self.getListNextStatesW(currentState):
            if self.worthExploring(son, depth + 1):
                # in state 'son', the first piece is the one just moved
                # we check which piece of currentState matches the one just moved
                if son[0][2] == currentState[0][2]:
                    movedPieceIndex = 0
                else:
                    movedPieceIndex = 1

                # move the piece to the new position
                self.chess.moveSim(currentState[movedPieceIndex], son[0])
                # recursive call with increased depth
                if self.DepthFirstSearchOptimized(son, depth + 1):
                    # If the method returns True, this means there was a checkmate
                    # add the state to the pathToTarget
                    self.pathToTarget.insert(0, currentState)
                    return True
                # restore the board to its previous state
                self.chess.moveSim(son[0], currentState[movedPieceIndex])


    def BreadthFirstSearch(self, currentState, depth):
        """
        Checkmate from currentStateW
        """
        BFSQueue = queue.Queue()
        # The root node has no parent, thus we add None, and -1 as the parent's depth
        self.dictPath[str(currentState)] = (None, -1)
        depthCurrentState = 0
        BFSQueue.put(currentState)
        self.listVisitedStates.append(currentState)
        # iterate until there are no more candidate nodes
        while BFSQueue.qsize() > 0:
            # Get the next node
            node = BFSQueue.get()
            depthNode = self.dictPath[str(node)][1] + 1
            if depthNode > self.depthMax:
                break
            # If not the root node, move the pieces from the previous to the current state
            if depthNode > 0:
                self.movePieces(currentState, depthCurrentState, node, depthNode)

            if self.isCheckMate(node):
                # If it is checkmate, reconstruct the optimal path found
                self.reconstructPath(node, depthNode)
                break

            for son in self.getListNextStatesW(node):
                if not self.isVisited(son):
                    self.listVisitedStates.append(son)
                    BFSQueue.put(son)
                    self.dictPath[str(son)] = (node, depthNode)
            currentState = node
            depthCurrentState = depthNode
    
    # Q learning 

    def stateToString(self, whiteState):
        """
        Convert the white pieces' state to a string representation.

        Input:
        - whiteState (list): List representing the state of white pieces.

        Returns:
        - stringState (str): String representation of the white pieces' state.
        """
        wkState = self.getPieceState(whiteState, 6)
        wrState = self.getPieceState(whiteState, 2)
        stringState = str(wkState[0]) + "," + str(wkState[1]) + ","
        if wrState is not None:
            stringState += str(wrState[0]) + "," + str(wrState[1])

        return stringState


    def stringToState(self, stringWhiteState):
        """
        Convert a string representation of white pieces' state to a list.

        Input:
        - stringWhiteState (str): String representation of the white pieces' state.

        Returns:
        - whiteState (list): List representing the state of white pieces.
        """
        whiteState = []
        whiteState.append([int(stringWhiteState[0]), int(stringWhiteState[2]), 6])
        if len(stringWhiteState) > 4:
            whiteState.append([int(stringWhiteState[4]), int(stringWhiteState[6]), 2])

        return whiteState


    def reconstructPathQL(self, initialState):
        """
        Reconstruct the path of moves based on the initial state using Q-values.

        Input:
        - initialState (list): Initial state of the chessboard. eg [[7, 0, 2], [7, 4, 6]]

        Returns:
        - path (list): List of states representing the sequence of moves.
        """
        currentState = initialState
        currentString = self.stateToString(initialState)
        checkMate = False
        self.chess.board.print_board()

        # Add the initial state to the path
        path = [initialState]
        while not checkMate:
            currentDict = self.qTable[currentString]
            maxQ = -100000
            maxState = None

            # Check which is the next state with the highest Q-value
            for stateString in currentDict.keys():
                qValue = currentDict[stateString]
                if maxQ < qValue:
                    maxQ = qValue
                    maxState = stateString

            state = self.stringToState(maxState)
            # When we get it, add it to the path
            path.append(state)
            movement = self.getMovement(currentState, state)
            # Make the corresponding movement
            self.chess.move(movement[0], movement[1])
            self.chess.board.print_board()
            currentString = maxState
            currentState = state

            # When it gets to checkmate, the execution is over
            if self.isCheckMate(state):
                checkMate = True

        print("Sequence of moves: ", path)
        
        return path
        
    def choose_action(self, state, epsilon, bk=None):
        possible_actions = self.getListNextStatesW(state)
        if not possible_actions:
            return None
        
        state_key = self.stateToString(state)
        explore = np.random.random() < epsilon
        
        best_action = None
        best_q_value = float('-inf')
        valid = []
        
        for action in possible_actions:
            piece_dest = action[0]
            
            if bk and piece_dest[0] == bk[0] and piece_dest[1] == bk[1]:
                continue
            
            valid.append(action)
            if not explore:
                action_key = self.stateToString(action)
                q_value = self.qTable[state_key][action_key]
                if q_value > best_q_value:
                    best_q_value = q_value
                    best_action = action
        if not valid:
            print("No valid actions available from state:", state)
            return None
        if explore:
            best_action = valid[np.random.randint(len(valid))]
        return best_action
        
    def nextStateQL(self, state, action):
        movement = self.getMovement(state, action)
        if movement[0] is None or movement[1] is None:
            print(f"Invalid movement: {state} -> {action}")
            return self.copyState(state)
        self.chess.moveSim(movement[0], movement[1], verbose=False)
        
        return self.getWhiteState(self.getCurrentStateSim())
    
    def getReward(self, state):
        if self.isCheckMate(state):
            return 100
        else:
            return -1
    
    def getQValue(self, state, action):
        state_key = self.stateToString(state)
        action_key = self.stateToString(action)
        return self.qTable[state_key][action_key]
    
    def setQValue(self, state, action, value):
        state_key = self.stateToString(state)
        action_key = self.stateToString(action)
        self.qTable[state_key][action_key] = value
        
    def trainQL(self, initialState, epochs, alpha, gamma, 
                epsilon, epsilon_min = 0.01, epsilon_decay = 0.995,
                convThreshold=0.001, patience=10):
        stable_epochs = 0
        bk = self.getPieceState(self.getCurrentState(), 12)  # Black king position
        for epoch in trange(epochs):
            currentState = self.copyState(initialState)
            self.newBoardSim(self.getCurrentState())
            oldQTable = copy.deepcopy(self.qTable)

            while not self.isCheckMate(currentState):
                oldState = self.copyState(currentState)
                action = self.choose_action(currentState, epsilon, bk)
                if action is None:
                    #print("No possible actions from state:", currentState)
                    break
                
                currentState = self.nextStateQL(currentState, action)
                reward = self.getReward(currentState)
                oldQValue = self.getQValue(oldState, action)
                
                temporalDifference = reward + (gamma * max(self.qTable[self.stateToString(currentState)].values(), default=0)) - oldQValue
                newQValue = oldQValue + (alpha * temporalDifference)
                self.setQValue(oldState, action, newQValue)
                
            deltaQ = 0.0
            all_states = set(list(self.qTable.keys()) + list(oldQTable.keys()))
            for state_key in all_states:
                all_actions = set(list(self.qTable[state_key].keys()) + list(oldQTable[state_key].keys()))
                for action_key in all_actions:
                    q_new = self.qTable[state_key].get(action_key, 0.0)
                    q_old = oldQTable[state_key].get(action_key, 0.0)
                    deltaQ += (q_new - q_old) ** 2
                
            deltaQ = np.sqrt(deltaQ)
            epsilon = max(epsilon * epsilon_decay, epsilon_min)
            if epoch % 100 == 0:
                print(f"\nEpoch {epoch}: DeltaQ={deltaQ:.6f}, Epsilon={epsilon:.4f}")
            if deltaQ < convThreshold:
                stable_epochs += 1
                if stable_epochs >= patience:
                    print(f"Converged after {epoch+1} epochs.")
                    break
            else:
                stable_epochs = 0
                
        print("Training completed.")
        print(f"Explored states: {len(self.qTable)}")
        
    def resetQL(self, TA):
        self.qTable = defaultdict(lambda: defaultdict(float))
        self.chess = chess.Chess(TA, True)
        

def print_menu():
    print("\n" + "="*40)
    print("       Q-Learning Chess - Menu")
    print("="*40)
    print("1. Print board")
    print("2. Train Q-Learning")
    print("3. Reconstruct path")
    print("4. Exit")
    print("="*40)

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit(usage())

    # Initialize an empty 8x8 chess board
    TA = np.zeros((8, 8))

    # Load initial positions of the pieces
    # White pieces
    TA[7][0] = 2  
    TA[7][5] = 6   
    TA[0][5] = 12  

    # Initialize AI chess with the board
    print("Starting AI chess...")
    aichess = Aichess(TA, True)
    
    # Get initial state
    currentState = aichess.chess.board.currentStateW
    print("Current State:", currentState, "\n")
    trained = False

    while True:
        print_menu()
        option = input("Select an option: ").strip()
        
        if option == "1":
            print("\n--- Current board ---")
            aichess.chess.board.print_board()
            
        elif option == "2":
            print("\n--- Training Q-Learning ---")
            # Reset board and Q-table for fresh training
            if trained:
                aichess.resetQL(TA)
                currentState = aichess.chess.boardSim.currentStateW
                
            aichess.trainQL(
                initialState=currentState, 
                epochs=10000, 
                alpha=0.1, 
                gamma=0.95, 
                epsilon=0.8, 
                epsilon_decay=0.995, 
                convThreshold=0.001, 
                patience=20 
            )
            trained = True
            print("Training completed.")
            
        elif option == "3":
            if not trained:
                print("\n[!] You must train the model first (option 2).")
            else:
                print("\n--- Reconstructing path ---")
                # Reset board to initial state
                aichess.chess = chess.Chess(TA, True)
                currentState = aichess.chess.board.currentStateW
                path = aichess.reconstructPathQL(currentState)
                print(f"\nPath length: {len(path)}")
                
        elif option == "4":
            print("\nGoodbye!")
            break
            
        else:
            print("\n[!] Invalid option. Try again.")
