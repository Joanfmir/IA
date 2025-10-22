#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:22:03 2022

@author: ignasi
"""
from collections import deque
import copy
import math

import chess
import board
import numpy as np
import sys
import queue
from typing import List

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
        List of all visited states during A* and other search algorithms.

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

    getNextPositions(state) -> list
        Returns a list of possible next positions for a given state.

    heuristica(currentState, color) -> int
        Calculates a heuristic value for the current state from the perspective of the given color.

    movePieces(start, depthStart, to, depthTo) -> None
        Moves all pieces along the path between two states.

    changeState(start, to) -> None
        Moves a single piece from start state to to state.

    reconstructPath(state, depth) -> None
        Reconstructs the path from initial state to the target state for A*.

    isWatchedWk(currentState) / isWatchedBk(currentState) -> bool
        Checks if the white or black king is under threat.

    allWkMovementsWatched(currentState) / allBkMovementsWatched(currentState) -> bool
        Checks if all moves of the white or black king are under threat.

    isWhiteInCheckMate(currentState) / isBlackInCheckMate(currentState) -> bool
        Determines if the white or black king is in checkmate.

    minimaxGame(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a full game using the Minimax algorithm for both white and black.

    alphaBetaPoda(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a game where both players use Minimax with Alpha-Beta Pruning.

    expectimax(depthWhite: int, depthBlack: int) -> To be implemented by you
        Simulates a full game where both players use the Expectimax algorithm.

    mean(values: list[float]) -> float
        Returns the arithmetic mean (average) of a list of numerical values.

    standardDeviation(values: list[float], mean_value: float) -> float
        Computes the standard deviation of a list of numerical values based on the given mean.

    calculateValue(values: list[float]) -> float
        Computes the expected value from a set of scores using soft-probabilities 
        derived from normalized values (exponential weighting). Can be useful for Expectimax.

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
        self.state_history = {}
        self.recent_states = deque(maxlen=4)
        self.rep_penalty =  10

    def state_to_key(self, state):
        sorted_state = sorted(tuple(piece) for piece in state if piece is not None)
        return tuple(sorted_state)
    
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

    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkPosition = self.getPieceState(currentState, 12)[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # If the white king has been captured, this is not a valid configuration
        if wkState is None:
            return False

        # Check all possible moves of the white king to see if it can capture the black king
        for wkPosition in self.getNextPositions(wkState):
            if bkPosition == wkPosition:
                # Black king would be in check
                return True

        if wrState is not None:
            # Check all possible moves of the white rook to see if it can capture the black king
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True

        return False

    def allBkMovementsWatched(self, currentState):
        # In this method, we check if the black king is threatened by the white pieces

        self.newBoardSim(currentState)
        # Get the current state of the black king
        bkState = self.getPieceState(currentState, 12)
        allWatched = False

        # If the black king is on the edge of the board, all its moves might be under threat
        if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
            wrState = self.getPieceState(currentState, 2)
            whiteState = self.getWhiteState(currentState)
            allWatched = True
            # Get the future states of the black pieces
            nextBStates = self.getListNextStatesB(self.getBlackState(currentState))

            for state in nextBStates:
                newWhiteState = whiteState.copy()
                # Check if the white rook has been captured; if so, remove it from the state
                if wrState is not None and wrState[0:2] == state[0][0:2]:
                    newWhiteState.remove(wrState)
                state = state + newWhiteState
                # Move the black pieces to the new state
                self.newBoardSim(state)

                # Check if in this position the black king is not threatened; 
                # if so, not all its moves are under threat
                if not self.isWatchedBk(state):
                    allWatched = False
                    break

        # Restore the original board state
        self.newBoardSim(currentState)
        return allWatched

    def isBlackInCheckMate(self, currentState):
        if self.isWatchedBk(currentState) and self.allBkMovementsWatched(currentState):
            return True

        return False


    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)

        wkPosition = self.getPieceState(currentState, 6)[0:2]
        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)

        # If the black king has been captured, this is not a valid configuration
        if bkState is None:
            return False

        # Check all possible moves for the black king and see if it can capture the white king
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                # White king would be in check
                return True

        if brState is not None:
            # Check all possible moves for the black rook and see if it can capture the white king
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False

    def allWkMovementsWatched(self, currentState):

        self.newBoardSim(currentState)
        # In this method, we check if the white king is threatened by black pieces
        # Get the current state of the white king
        wkState = self.getPieceState(currentState, 6)
        allWatched = False

        # If the white king is on the edge of the board, it may be more vulnerable
        if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
            # Get the state of the black pieces
            brState = self.getPieceState(currentState, 8)
            blackState = self.getBlackState(currentState)
            allWatched = True

            # Get the possible future states for the white pieces
            nextWStates = self.getListNextStatesW(self.getWhiteState(currentState))
            for state in nextWStates:
                newBlackState = blackState.copy()
                # Check if the black rook has been captured. If so, remove it from the state
                if brState is not None and brState[0:2] == state[0][0:2]:
                    newBlackState.remove(brState)
                state = state + newBlackState
                # Move the white pieces to their new state
                self.newBoardSim(state)
                # Check if the white king is not threatened in this position,
                # which implies that not all of its possible moves are under threat
                if not self.isWatchedWk(state):
                    allWatched = False
                    break

        # Restore the original board state
        self.newBoardSim(currentState)
        return allWatched


    def isWhiteInCheckMate(self, currentState):
        if self.isWatchedWk(currentState) and self.allWkMovementsWatched(currentState):
            return True
        return False
    

    def heuristica(self, currentState, color):
        # This method calculates the heuristic value for the current state.
        # The value is initially computed from White's perspective.
        # If the 'color' parameter indicates Black, the final value is multiplied by -1.

        value = 0

        bkState = self.getPieceState(currentState, 12)  # Black King
        wkState = self.getPieceState(currentState, 6)   # White King
        wrState = self.getPieceState(currentState, 2)   # White Rook
        brState = self.getPieceState(currentState, 8)   # Black Rook

        filaBk, columnaBk = bkState[0], bkState[1]
        filaWk, columnaWk = wkState[0], wkState[1]

        if wrState is not None:
            filaWr, columnaWr = wrState[0], wrState[1]
        if brState is not None:
            filaBr, columnaBr = brState[0], brState[1]

        # If the black rook has been captured
        if brState is None:
            value += 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and wrState is not None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                value += (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            # For White: the closer our king is to the opponent’s king, the better.
            # Subtract 7 from the king-to-king distance since 7 is the maximum distance possible on the board.
            value += (7 - distReis)

            # If the black king is against a wall, prioritize pushing him into a corner (ideal for checkmate).
            if bkState[0] in (0, 7) or bkState[1] in (0, 7):
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            # Otherwise, encourage moving the black king closer to the wall.
            else:
                value += (max(abs(filaBk - 3.5), abs(columnaBk - 3.5))) * 10

        # If the white rook has been captured.
        # The logic is similar to the previous section but with reversed (negative) values.
        if wrState is None:
            value -= 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and brState is not None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                value -= (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            # For White: being closer to the opposing king is better.
            # Subtract 7 from the distance since that’s the maximum possible distance.
            value += (-7 + distReis)

            # If the white king is against a wall, penalize that position.
            if wkState[0] in (0, 7) or wkState[1] in (0, 7):
                value -= (abs(filaWk - 3.5) + abs(columnaWk - 3.5)) * 10
            # Otherwise, encourage the king to stay away from the wall.
            else:
                value -= (max(abs(filaWk - 3.5), abs(columnaWk - 3.5))) * 10

        # If the black king is in check, reward this state.
        if self.isWatchedBk(currentState):
            value += 20

        # If the white king is in check, penalize this state.
        if self.isWatchedWk(currentState):
            value -= 20

        #state_key = self.state_to_key(currentState)

        #recent_reps = sum(1 for recent_key in self.recent_states if recent_key == state_key)
        #value -= recent_reps * self.rep_penalty


        # If the current player is Black, invert the heuristic value.
        if not color:
            value *= -1

        return value
    
    def mean(self, values):
        # Calculate the arithmetic mean (average) of a list of numeric values.
        total = 0
        n = len(values)
        
        for i in range(n):
            total += values[i]

        return total / n


    def standard_deviation(self, values, mean_value):
        # Calculate the standard deviation of a list of values.
            total = 0
            n = len(values)

            for i in range(n):
                total += pow(values[i] - mean_value, 2)

            return pow(total / n, 1 / 2)


    def calculateValue(self, values):
        # Calculate a weighted expected value based on normalized probabilities. - useful for Expectimax.
        
        # Compute mean and standard deviation
        mean_value = self.mean(values)
        std_dev = self.standard_deviation(values, mean_value)

        # If all values are equal, the deviation is 0, equal probability
        if std_dev == 0:
            return values[0]

        expected_value = 0
        total_weight = 0
        n = len(values)

        for i in range(n):
            # Normalize value using z-score
            normalized_value = (values[i] - mean_value) / std_dev

            # Convert to a positive weight using e^(-x)
            positive_weight = pow(1 / math.e, normalized_value)

            # Weighted sum
            expected_value += positive_weight * values[i]
            total_weight += positive_weight

        # Final expected value (weighted average)
        return expected_value / total_weight

    WHITES = True
    BLACKS = False
    '''
    def minimaxGame(self, depthWhite, depthBlack):

        currentState = self.getCurrentState()

        turn = self.WHITES
        num_moves = 0

        while (not self.isWhiteInCheckMate(currentState) and not self.isBlackInCheckMate(currentState)) and num_moves < 20:
            if turn:
                _, nextMove = self.max_recursion(currentState, depthWhite, self.WHITES, turn)
                print("White's move:", nextMove)
                self.chess.move(*nextMove) 
            else:
                _, nextMove = self.max_recursion(currentState, depthBlack, self.BLACKS, turn)
                print("Black's move:", nextMove)
                self.chess.move(*nextMove)
            turn = not turn
            num_moves += 1
            currentState = self.getCurrentState()
            self.chess.board.print_board()
            if len(currentState) <= 2:
                print("Game over")
                break
    '''

    def minimaxGame(self, depthWhite, depthBlack):
        """
        Simula un juego completo usando Minimax para ambos jugadores.
        """
        currentState = self.getCurrentState()
        turn = self.WHITES
        num_moves = 0

        print("Iniciando juego Minimax...")
        print(f"Profundidad Blancas: {depthWhite}, Profundidad Negras: {depthBlack}")
        
        #while (not self.is_checkmate(currentState, self.WHITES) and 
        #    not self.is_checkmate(currentState, self.BLACKS)) and num_moves < 100:
        while (not self.isWhiteInCheckMate(currentState) and 
            not self.isBlackInCheckMate(currentState)) and num_moves < 100:

            print(f"\n--- Movimiento {num_moves + 1} ---")
            
            if turn == self.WHITES:
                eval_score, nextMove = self._minimax_alphaBeta(currentState, depthWhite, self.WHITES, turn)
                print(f"Evaluación Blancas: {eval_score}")
                print(f"White's move: {nextMove}")
                
                if nextMove is None:
                    print("White has no valid moves - Game Over")
                    break
                self.chess.move(*nextMove)
            else:
                eval_score, nextMove = self._minimax_alphaBeta(currentState, depthBlack, self.BLACKS, turn)
                print(f"Evaluación Negras: {eval_score}")
                print(f"Black's move: {nextMove}")
                
                if nextMove is None:
                    print("Black has no valid moves - Game Over")
                    break
                self.chess.move(*nextMove)

            turn = not turn
            num_moves += 1
            currentState = self.getCurrentState()

            #state_key = self.state_to_key(currentState)
            #self.state_history[state_key] = self.state_history.get(state_key, 0) + 1
            #self.recent_states.append(state_key)
            
            print("\nTablero actual:")
            self.chess.board.print_board()
            
            # Verificar si solo quedan los reyes
            if len(currentState) <= 2:
                print("Solo quedan los reyes - Game over")
                break
        
        # Resultado final
        print("\n" + "="*50)
        if self.isWhiteInCheckMate(currentState):
            print("¡JAQUE MATE! Negras ganan")
        elif self.isBlackInCheckMate(currentState):
            print("¡JAQUE MATE! Blancas ganan")
        else:
            print(f"Juego terminado después de {num_moves} movimientos")
        print("="*50)
    
    '''
    def _minimax(self, state, depth, max_color, turn):
        self.newBoardSim(state)
        if depth <= 0 or self.is_checkmate(state, self.WHITES) or self.is_checkmate(state, self.BLACKS):
            if depth > 0:
               print(f"Jaque mate detectado.")
            return self.heuristica(state, max_color), None

        self.newBoardSim(state)
        if turn:
            max_eval = -math.inf
            max_move = None
            
            nextMoves = [self.getMovement(state, newState) for newState in self.getListNextStatesW(state)]
            for move in nextMoves:
                if None in move:
                    print("Movimiento inválido encontrado, se omite.")
                    continue
                self.newBoardSim(state)
                self.chess.moveSim(*move, verbose=False)
                if self.isWatchedWk(self.getCurrentStateSim()):
                    continue
                eval, _ = self._minimax(self.getCurrentStateSim(), depth - 1, max_color, not turn)
                if eval > max_eval:
                    max_eval = eval
                    max_move = move
            if max_move is None:
                return self.heuristica(state, max_color), None
            return max_eval, max_move
        else:
            min_eval = math.inf
            min_move = None

            nextMoves = [self.getMovement(state, newState) for newState in self.getListNextStatesB(state)]
            for move in nextMoves:
                if None in move:
                    print("Movimiento inválido encontrado, se omite.")
                    continue
                self.newBoardSim(state)
                self.chess.moveSim(*move, verbose=False)
                if self.isWatchedBk(self.getCurrentStateSim()):
                    continue
                eval, _ = self._minimax(self.getCurrentStateSim(), depth - 1, max_color, not turn)
                self.newBoardSim(state)
                if eval < min_eval:
                    min_eval = eval
                    min_move = move
            if min_move is None:
                return self.heuristica(state, max_color), None
            return min_eval, min_move
    '''
    '''
    def _minimax(self, state, depth, max_color, turn):
        """
        Variante del minimax que filtra los movimientos que dejan en jaque al propio rey
        y evita las llamadas redundantes a newBoardSim.
        """
        self.newBoardSim(state)
        if depth <= 0 or self.is_checkmate(state, self.WHITES) or self.is_checkmate(state, self.BLACKS):
            return self.heuristica(state, max_color), None
        maximizing = (turn == max_color)
        self.newBoardSim(state)
        if turn:
            next_states = self.getListNextStatesW(state)
            best_eval = -math.inf
        else:
            next_states = self.getListNextStatesB(state)
            best_eval = math.inf
        best_move = None
        for new_state in next_states:
            move = self.getMovement(state, new_state)
            if None in move:
                continue
            self.newBoardSim(state)
            self.chess.moveSim(*move, verbose=False)
            sim_state = self.getCurrentStateSim()
            if turn and self.isWatchedWk(sim_state):
                continue
            if not turn and self.isWatchedBk(sim_state):
                continue
            eval_score, _ = self._minimax(sim_state, depth - 1, max_color, not turn)
            self.newBoardSim(state)
            if maximizing:
                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            else:
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
        if best_move is None:
            print("No hay movimientos válidos, se aplica heurística.")
            return self.heuristica(state, max_color), None
        return best_eval, best_move
    '''
    
    def _minimax(self, state, depth, maximizing_player, turn):
        """
        Implementación de minimax puro.
        
        Args:
            state: Estado actual del tablero
            depth: Profundidad de búsqueda restante
            maximizing_player: WHITES o BLACKS - quién inició la búsqueda (NO cambia)
            turn: WHITES o BLACKS - de quién es el turno actual (CAMBIA en cada nivel)
        
        Returns:
            tuple: (evaluación, mejor_movimiento)
        """
        
        # Condiciones de parada
        #if depth <= 0 or self.is_checkmate(state, self.WHITES) or self.is_checkmate(state, self.BLACKS):
        #    return self.heuristica(state, maximizing_player), None

        if depth <= 0 or self.isWhiteInCheckMate(state) or self.isBlackInCheckMate(state):
            return self.heuristica(state, maximizing_player), None

        # Determinar si estamos maximizando o minimizando
        is_maximizing = (turn == maximizing_player)
        # TODO: Error en gerlistnextstateB que no identifica por color comentar en clase
        self.newBoardSim(state)
        # Obtener los próximos estados según el turno
        if turn == self.WHITES:
            nextStates = self.getListNextStatesW(state)
        else:
            nextStates = self.getListNextStatesB(state)
        # Validar movimientos y filtrar los que dejan al propio rey en jaque
        valid_moves = []
        for newState in nextStates:
            move = self.getMovement(state, newState)
            if None in move:
                continue
            
            # Simular el movimiento
            self.newBoardSim(state)
            self.chess.moveSim(*move, verbose=False)
            simState = self.getCurrentStateSim()
            
            # Verificar que el movimiento no deje al propio rey en jaque
            if turn == self.WHITES:
                if self.isWatchedWk(simState):
                    continue  # Movimiento inválido, deja al rey blanco en jaque
            else:
                if self.isWatchedBk(simState):
                    continue  # Movimiento inválido, deja al rey negro en jaque
            

            valid_moves.append((move, simState))

        self.newBoardSim(state)

        # Si no hay movimientos válidos
        if not valid_moves:
            if (turn == self.WHITES and self.isWatchedWk(state)) or \
            (turn == self.BLACKS and self.isWatchedBk(state)):
                return (-math.inf if turn == maximizing_player else math.inf, None)
            return 0, None  # Ahogado
        
        # Aplicar Minimax
        if is_maximizing:
            # Maximizando: buscar el mejor movimiento (mayor evaluación)
            best_eval = -math.inf
            best_move = None
            
            for move, new_state in valid_moves:
                # Llamada recursiva alternando el turno
                eval_score, _ = self._minimax(new_state, depth - 1, maximizing_player, not turn)

                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
            
            return best_eval, best_move
        
        else:  # Minimizando
            # Minimizando: buscar el movimiento que minimiza la evaluación
            best_eval = math.inf
            best_move = None
            
            for move, new_state in valid_moves:
                # Llamada recursiva alternando el turno
                eval_score, _ = self._minimax(new_state, depth - 1, maximizing_player, not turn)
                
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
            
            return best_eval, best_move

    def _minimax_alphaBeta(self, state, depth, maximizing_player, turn, alpha=-math.inf, beta=math.inf):
        """
        Implementación de minimax puro.
        
        Args:
            state: Estado actual del tablero
            depth: Profundidad de búsqueda restante
            maximizing_player: WHITES o BLACKS - quién inició la búsqueda (NO cambia)
            turn: WHITES o BLACKS - de quién es el turno actual (CAMBIA en cada nivel)
        
        Returns:
            tuple: (evaluación, mejor_movimiento)
        """
        
        # Condiciones de parada
        #if depth <= 0 or self.is_checkmate(state, self.WHITES) or self.is_checkmate(state, self.BLACKS):
        #    return self.heuristica(state, maximizing_player), None

        if depth <= 0 or self.isWhiteInCheckMate(state) or self.isBlackInCheckMate(state):
            return self.heuristica(state, maximizing_player), None

        # Determinar si estamos maximizando o minimizando
        is_maximizing = (turn == maximizing_player)
        # TODO: Error en gerlistnextstateB que no identifica por color comentar en clase
        self.newBoardSim(state)
        # Obtener los próximos estados según el turno
        if turn == self.WHITES:
            nextStates = self.getListNextStatesW(state)
        else:
            nextStates = self.getListNextStatesB(state)
        # Validar movimientos y filtrar los que dejan al propio rey en jaque
        valid_moves = []
        for newState in nextStates:
            move = self.getMovement(state, newState)
            if None in move:
                continue
            
            # Simular el movimiento
            self.newBoardSim(state)
            self.chess.moveSim(*move, verbose=False)
            simState = self.getCurrentStateSim()
            
            # Verificar que el movimiento no deje al propio rey en jaque
            if turn == self.WHITES:
                if self.isWatchedWk(simState):
                    continue  # Movimiento inválido, deja al rey blanco en jaque
            else:
                if self.isWatchedBk(simState):
                    continue  # Movimiento inválido, deja al rey negro en jaque
            

            valid_moves.append((move, simState))

        self.newBoardSim(state)

        # Si no hay movimientos válidos
        if not valid_moves:
            if (turn == self.WHITES and self.isWatchedWk(state)) or \
            (turn == self.BLACKS and self.isWatchedBk(state)):
                return (-math.inf if turn == maximizing_player else math.inf, None)
            return 0, None  # Ahogado
        
        # Aplicar Minimax
        if is_maximizing:
            # Maximizando: buscar el mejor movimiento (mayor evaluación)
            best_eval = -math.inf
            best_move = None
            
            for move, new_state in valid_moves:
                # Llamada recursiva alternando el turno
                eval_score, _ = self._minimax_alphaBeta(new_state, depth - 1, maximizing_player, not turn, alpha, beta)

                if eval_score > best_eval:
                    best_eval = eval_score
                    best_move = move
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break
            
            return best_eval, best_move
        
        else:  # Minimizando
            # Minimizando: buscar el movimiento que minimiza la evaluación
            best_eval = math.inf
            best_move = None
            
            for move, new_state in valid_moves:
                # Llamada recursiva alternando el turno
                eval_score, _ = self._minimax_alphaBeta(new_state, depth - 1, maximizing_player, not turn, alpha, beta)
                
                if eval_score < best_eval:
                    best_eval = eval_score
                    best_move = move
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break
            
            return best_eval, best_move
     

    def getCurrentStateSim(self):
        listStates = []
        for i in self.chess.boardSim.currentStateW:
            listStates.append(i)
        for j in self.chess.boardSim.currentStateB:
            listStates.append(j)
        return listStates

    def alphaBetaPoda(self, depthWhite,depthBlack):
        
        currentState = self.getCurrentState()
        # Your code here  
        
    def expectimax(self, depthWhite, depthBlack):
        
        currentState = self.getCurrentState()
        # Your code here       
        

if __name__ == "__main__":
    # if len(sys.argv) < 2:
    #     sys.exit(usage())

    # Initialize an empty 8x8 chess board
    TA = np.zeros((8, 8))


    # Load initial positions of the pieces
    TA = np.zeros((8, 8))
    TA[7][0] = 2   
    TA[7][3] = 6   
    TA[0][7] = 8   
    TA[0][3] = 12  

    # Initialise board and print
    print("stating AI chess... ")
    aichess = Aichess(TA, True)
    print("printing board")
    aichess.chess.boardSim.print_board()
    
    # Run exercise 1
    aichess.minimaxGame(4,4)
    # Add code to save results and continue with other exercises
