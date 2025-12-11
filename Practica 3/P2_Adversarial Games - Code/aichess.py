#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Práctica 3 - Aichess con Q-learning para el primer tablero de P1.

Escenario: rey negro, rey blanco y torre blanca. Solo mueven las blancas.
Se usa Q-learning para aprender a dar mate.
"""

import math
from itertools import permutations
from typing import List

import numpy as np

import chess
import board

RawStateType = List[List[List[int]]]


class Aichess():
    """
    Clase envoltorio sobre el simulador de ajedrez para:
    - Manejar estados (listas [fila, col, pieza])
    - Detectar jaque y jaque mate
    - Aplicar una heurística
    - Aplicar Q-learning en el escenario rey + torre vs rey
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
        self.depthMax = 8
        # Dictionary to reconstruct the visited path (P1 – búsqueda)
        self.dictPath = {}
        # Dictionary to control visited states (DFS optimized)
        self.dictVisitedStates = {}

        # Q-table para Q-learning:
        # clave: string del estado de BLANCAS (wk, wr)
        # valor: diccionario {string_siguiente_estado: Q}
        self.qTable = {}

        # Guardamos el estado inicial completo (blancas + negras) para reusar
        self.initialFullState = self.getCurrentState()

    # ------------------------------------------------------------------
    #  Funciones auxiliares sobre estados
    # ------------------------------------------------------------------

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
        # Crea un nuevo boardSim a partir de una lista de piezas
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
        # Estado actual completo (blancas + negras) del tablero REAL
        listStates = []
        for i in self.chess.board.currentStateW:
            listStates.append(i)
        for j in self.chess.board.currentStateB:
            listStates.append(j)
        return listStates

    def getNextPositions(self, state):
        # Dada una pieza (state), devuelve posiciones [fila, col] alcanzables
        if state is None:
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
        # Extrae del estado completo las piezas blancas (rey y torre)
        whiteState = []
        wkState = self.getPieceState(currentState, 6)
        whiteState.append(wkState)
        wrState = self.getPieceState(currentState, 2)
        if wrState is not None:
            whiteState.append(wrState)
        return whiteState

    def getBlackState(self, currentState):
        # Extrae del estado completo las piezas negras
        blackState = []
        bkState = self.getPieceState(currentState, 12)
        blackState.append(bkState)
        brState = self.getPieceState(currentState, 8)
        if brState is not None:
            blackState.append(brState)
        return blackState

    def getMovement(self, state, nextState):
        # Dado un estado y un sucesor, devuelve [pieza_inicial, pieza_final] movida
        pieceState = None
        pieceNextState = None
        for piece in state:
            if piece not in nextState:
                movedPiece = piece[2]
                pieceNext = self.getPieceState(nextState, movedPiece)
                if pieceNext is not None:
                    pieceState = piece
                    pieceNextState = pieceNext
                    break

        return [pieceState, pieceNextState]

    # ------------------------------------------------------------------
    #  Detección de jaque y jaque mate
    # ------------------------------------------------------------------

    def isWatchedBk(self, currentState):

        self.newBoardSim(currentState)

        bkPosition = self.getPieceState(currentState, 12)[0:2]
        wkState = self.getPieceState(currentState, 6)
        wrState = self.getPieceState(currentState, 2)

        # Si el rey blanco ha sido capturado, no es una configuración válida
        if wkState is None:
            return False

        # ¿Puede el rey blanco capturar al rey negro?
        for wkPosition in self.getNextPositions(wkState):
            if bkPosition == wkPosition:
                return True

        # ¿Puede la torre blanca capturar al rey negro?
        if wrState is not None:
            for wrPosition in self.getNextPositions(wrState):
                if bkPosition == wrPosition:
                    return True

        return False

    def allBkMovementsWatched(self, currentState):
        # Comprueba si todos los movimientos del rey negro están bajo amenaza

        self.newBoardSim(currentState)
        bkState = self.getPieceState(currentState, 12)
        allWatched = False

        if bkState[0] == 0 or bkState[0] == 7 or bkState[1] == 0 or bkState[1] == 7:
            wrState = self.getPieceState(currentState, 2)
            whiteState = self.getWhiteState(currentState)
            allWatched = True
            nextBStates = self.getListNextStatesB(self.getBlackState(currentState))

            for state in nextBStates:
                newWhiteState = whiteState.copy()
                # Si el rey negro captura la torre blanca
                if wrState is not None and wrState[0:2] == state[0][0:2]:
                    newWhiteState.remove(wrState)
                state = state + newWhiteState
                self.newBoardSim(state)

                if not self.isWatchedBk(state):
                    allWatched = False
                    break

        self.newBoardSim(currentState)
        return allWatched

    def isBlackInCheckMate(self, currentState):
        # Jaque mate al rey negro
        if self.isWatchedBk(currentState) and self.allBkMovementsWatched(currentState):
            return True
        return False

    def isWatchedWk(self, currentState):
        self.newBoardSim(currentState)

        wkPosition = self.getPieceState(currentState, 6)[0:2]
        bkState = self.getPieceState(currentState, 12)
        brState = self.getPieceState(currentState, 8)

        if bkState is None:
            return False

        # ¿Puede el rey negro capturar al rey blanco?
        for bkPosition in self.getNextPositions(bkState):
            if wkPosition == bkPosition:
                return True

        # ¿Puede la torre negra capturar al rey blanco?
        if brState is not None:
            for brPosition in self.getNextPositions(brState):
                if wkPosition == brPosition:
                    return True

        return False

    def allWkMovementsWatched(self, currentState):

        self.newBoardSim(currentState)
        wkState = self.getPieceState(currentState, 6)
        allWatched = False

        if wkState[0] == 0 or wkState[0] == 7 or wkState[1] == 0 or wkState[1] == 7:
            brState = self.getPieceState(currentState, 8)
            blackState = self.getBlackState(currentState)
            allWatched = True

            nextWStates = self.getListNextStatesW(self.getWhiteState(currentState))
            for state in nextWStates:
                newBlackState = blackState.copy()
                if brState is not None and brState[0:2] == state[0][0:2]:
                    newBlackState.remove(brState)
                state = state + newBlackState
                self.newBoardSim(state)
                if not self.isWatchedWk(state):
                    allWatched = False
                    break

        self.newBoardSim(currentState)
        return allWatched

    def isWhiteInCheckMate(self, currentState):
        if self.isWatchedWk(currentState) and self.allWkMovementsWatched(currentState):
            return True
        return False

    # Para Q-learning: checkmate = mate al rey negro
    def isCheckMate(self, state_unused=None):
        """
        Wrapper para usar en Q-learning.
        Ignora el parámetro y mira el tablero actual (boardSim/board).
        """
        current = self.getCurrentState()
        return self.isBlackInCheckMate(current)

    # ------------------------------------------------------------------
    #  Heurística (la misma que ya tenías)
    # ------------------------------------------------------------------

    def heuristica(self, currentState, color):
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

        # Black rook captured
        if brState is None:
            value += 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and wrState is not None:
                filaR = abs(filaBk - filaWr)
                columnaR = abs(columnaWr - columnaBk)
                value += (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            value += (7 - distReis)

            if bkState[0] in (0, 7) or bkState[1] in (0, 7):
                value += (abs(filaBk - 3.5) + abs(columnaBk - 3.5)) * 10
            else:
                value += (max(abs(filaBk - 3.5), abs(columnaBk - 3.5))) * 10

        # White rook captured
        if wrState is None:
            value -= 50
            fila = abs(filaBk - filaWk)
            columna = abs(columnaWk - columnaBk)
            distReis = min(fila, columna) + abs(fila - columna)

            if distReis >= 3 and brState is not None:
                filaR = abs(filaWk - filaBr)
                columnaR = abs(columnaBr - columnaWk)
                value -= (min(filaR, columnaR) + abs(filaR - columnaR)) / 10

            value += (-7 + distReis)

            if wkState[0] in (0, 7) or wkState[1] in (0, 7):
                value -= (abs(filaWk - 3.5) + abs(columnaWk - 3.5)) * 10
            else:
                value -= (max(abs(filaWk - 3.5), abs(columnaWk - 3.5))) * 10

        if self.isWatchedBk(currentState):
            value += 20

        if self.isWatchedWk(currentState):
            value -= 20

        if not color:
            value *= -1

        return value

    # ------------------------------------------------------------------
    #  Estadística para Expectimax (las usa calculateValue si quieres)
    # ------------------------------------------------------------------

    def mean(self, values):
        total = 0
        n = len(values)
        for i in range(n):
            total += values[i]
        return total / n

    def standard_deviation(self, values, mean_value):
        total = 0
        n = len(values)
        for i in range(n):
            total += pow(values[i] - mean_value, 2)
        return pow(total / n, 1 / 2)

    def calculateValue(self, values):
        mean_value = self.mean(values)
        std_dev = self.standard_deviation(values, mean_value)

        if std_dev == 0:
            return values[0]

        expected_value = 0
        total_weight = 0
        n = len(values)

        for i in range(n):
            normalized_value = (values[i] - mean_value) / std_dev
            positive_weight = pow(1 / math.e, normalized_value)
            expected_value += positive_weight * values[i]
            total_weight += positive_weight

        return expected_value / total_weight

    # ------------------------------------------------------------------
    #  MÉTODOS DEL PROFESOR (auxiliary_P3)
    # ------------------------------------------------------------------

    def stateToString(self, whiteState):
        """
        Convierte el estado de las piezas BLANCAS a string.
        Formato: "fila_wk,col_wk[,fila_wr,col_wr]"
        """
        wkState = self.getPieceState(whiteState, 6)
        wrState = self.getPieceState(whiteState, 2)
        stringState = str(wkState[0]) + "," + str(wkState[1]) + ","
        if wrState is not None:
            stringState += str(wrState[0]) + "," + str(wrState[1])

        return stringState

    def stringToState(self, stringWhiteState):
        """
        Convierte un string de estado de BLANCAS a lista de piezas.
        """
        whiteState = []
        # wk siempre está
        whiteState.append([int(stringWhiteState[0]), int(stringWhiteState[2]), 6])
        # si hay torre, el string es largo: "r,c,r2,c2"
        if len(stringWhiteState) > 4:
            whiteState.append([int(stringWhiteState[4]), int(stringWhiteState[6]), 2])

        return whiteState

    def reconstructPath(self, initialState):
        """
        Reconstruye la secuencia de jugadas basándose en la Q-table.

        initialState: estado inicial COMPLETO (incluyendo negras), ej:
            [[7, 0, 2], [7, 4, 6], [0, 5, 12]]

        Usa solo el estado de BLANCAS para indexar la Q-table.
        """
        # Cogemos solo blancas para indexar Q-table
        currentWhiteState = self.getWhiteState(initialState)
        currentString = self.stateToString(currentWhiteState)
        checkMate = False

        # Aseguramos que el tablero real esté en el estado inicial
        self.newBoardSim(initialState)
        self.chess.board = board.Board(np.zeros((8, 8)))
        for p in initialState:
            self.chess.board.addPiece(p)
        self.chess.board.print_board()

        path = [currentWhiteState]

        while not checkMate:
            currentDict = self.qTable[currentString]
            maxQ = -100000
            maxState = None

            # Elegimos el siguiente estado (acción) con mayor Q
            for stateString in currentDict.keys():
                qValue = currentDict[stateString]
                if maxQ < qValue:
                    maxQ = qValue
                    maxState = stateString

            nextWhiteState = self.stringToState(maxState)
            path.append(nextWhiteState)

            # Movimiento real en el tablero (Board) y visualización
            fullCurrent = self.getCurrentState()
            fullNext = nextWhiteState + self.getBlackState(fullCurrent)
            movement = self.getMovement(fullCurrent, fullNext)
            self.chess.move(movement[0], movement[1])
            self.chess.board.print_board()

            currentString = maxState
            currentWhiteState = nextWhiteState

            if self.isCheckMate(None):
                checkMate = True

        print("Sequence of moves (white states): ", path)

    # ------------------------------------------------------------------
    #  Q-LEARNING PARA EL ESCENARIO DE LA PRÁCTICA 3
    # ------------------------------------------------------------------

    def qLearningChess(self,
                       episodes: int,
                       alpha: float,
                       gamma: float,
                       epsilon: float,
                       use_heuristic_reward: bool = False,
                       stochastic: bool = False,
                       intended_prob: float = 1.0,
                       max_steps: int = 100):
        """
        Q-learning para el escenario:
        - rey negro, rey blanco, torre blanca
        - solo mueven las blancas
        - el estado para la Q-table solo incluye BLANCAS (wk, wr)

        Parámetros:
        - episodes: nº de partidas de entrenamiento
        - alpha: tasa de aprendizaje
        - gamma: factor de descuento
        - epsilon: prob. de exploración (ε-greedy)
        - use_heuristic_reward: si True, recompensa basada en heurística
                                si False, recompensa -1 y +100 en mate
        - stochastic: si True, "marinero borracho": a veces se ejecuta
                      una acción aleatoria en lugar de la elegida
        - intended_prob: prob. de que se ejecute la acción elegida
                         (cuando stochastic=True)
        - max_steps: límite de jugadas por episodio
        """

        # Estado completo inicial (blancas + negras)
        full_initial = self.initialFullState
        blackState_fixed = self.getBlackState(full_initial)

        for ep in range(episodes):
            # Reiniciar tablero de simulación al estado inicial
            self.newBoardSim(full_initial)

            whiteState = self.getWhiteState(full_initial)
            state_str = self.stateToString(whiteState)

            done = False
            steps = 0

            while not done and steps < max_steps:
                # Aseguramos entrada en Q-table
                if state_str not in self.qTable:
                    self.qTable[state_str] = {}

                # Lista de movimientos posibles de BLANCAS
                nextWhiteStates = []
                self.newBoardSim(whiteState + blackState_fixed)
                self.getListNextStatesW(whiteState)
                for ns in self.listNextStates:
                    nextWhiteStates.append(ns)

                if not nextWhiteStates:
                    # Sin movimientos legales: fin episodio
                    break

                # ---- SELECCIÓN DE ACCIÓN (ε-greedy) ----
                import random
                if random.random() < epsilon:
                    # Exploración: acción aleatoria
                    chosenNextWhite = random.choice(nextWhiteStates)
                else:
                    # Explotación: acción con mayor Q
                    best_q = -1e9
                    best_ns = None
                    for ns in nextWhiteStates:
                        ns_str = self.stateToString(ns)
                        q_val = self.qTable[state_str].get(ns_str, 0.0)
                        if q_val > best_q:
                            best_q = q_val
                            best_ns = ns
                    chosenNextWhite = best_ns

                # Si hay estocasticidad: a veces cambiamos a una acción random
                if stochastic and random.random() > intended_prob:
                    chosenNextWhite = random.choice(nextWhiteStates)

                next_state_str = self.stateToString(chosenNextWhite)

                # ---- CALCULO DE RECOMPENSA ----
                # Construimos estado completo para evaluar checkmate / heurística
                full_current = whiteState + blackState_fixed
                full_next = chosenNextWhite + blackState_fixed

                self.newBoardSim(full_next)

                if not use_heuristic_reward:
                    # Caso (a): recompensa -1 por movimiento y +100 por mate
                    if self.isBlackInCheckMate(full_next):
                        reward = 100.0
                        done = True
                    else:
                        reward = -1.0
                else:
                    # Caso (b): recompensa basada en heurística
                    h_current = self.heuristica(full_current, True)
                    h_next = self.heuristica(full_next, True)
                    # bono si mate, para mantener +100
                    if self.isBlackInCheckMate(full_next):
                        reward = 100.0
                        done = True
                    else:
                        reward = (h_next - h_current)

                # ---- ACTUALIZACIÓN Q(s,a) ----
                old_q = self.qTable[state_str].get(next_state_str, 0.0)

                # Max Q del siguiente estado (si existe)
                if next_state_str not in self.qTable:
                    max_q_next = 0.0
                else:
                    max_q_next = max(self.qTable[next_state_str].values(), default=0.0)

                new_q = old_q + alpha * (reward + gamma * max_q_next - old_q)
                self.qTable[state_str][next_state_str] = new_q

                # Pasamos al siguiente estado
                whiteState = chosenNextWhite
                state_str = next_state_str
                steps += 1

            # Pequeño print opcional para ver progreso
            # print(f"Episode {ep+1}/{episodes} finished in {steps} steps.")

        # Al terminar los episodios, la Q-table está entrenada.
        # La ruta óptima se puede reconstruir con reconstructPath().


if __name__ == "__main__":
    # Tablero inicial: ejemplo típico de rey + torre vs rey
    TA = np.zeros((8, 8))
    # Torre blanca
    TA[7][0] = 2
    # Rey blanco
    TA[7][5] = 6
    # Rey negro
    TA[0][5] = 12

    print("Starting AI chess with Q-learning scenario...")
    aichess = Aichess(TA, True)
    print("Initial board:")
    aichess.chess.board.print_board()

    # Ejemplo de entrenamiento simple (cosas pequeñas para probar):
    # aichess.qLearningChess(episodes=1000, alpha=0.1, gamma=0.95, epsilon=0.2)

    # Luego: aichess.reconstructPath(aichess.initialFullState)
