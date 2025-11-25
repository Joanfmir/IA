# Methods to add in aichess.py

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


def reconstructPath(self, initialState):
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