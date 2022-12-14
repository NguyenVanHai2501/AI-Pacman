# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        

        "*** YOUR CODE HERE ***"

        # V??ng l???p gi?? tr??? ???????c thi???t l???p cho c??c l???n l???p
        for i in range(self.iterations):
            # Khai b??o 1 b??? ?????m ????? l??u gi?? tr??? c???a l???n l???p cho m???i tr???ng th??i
            iterationValue = util.Counter()
            # l???p t???ng tr???ng th??i (state)
            for state in self.mdp.getStates():
                # n???u tr???ng th??i l?? tr???ng th??i k???t th??c th?? ph???n th?????ng l?? ph???n th?????ng tho??t
                # v?? kh??ng c?? ph???n th?????ng chi???t kh???u
                if self.mdp.isTerminal(state):
                    self.values[state] = self.mdp.getReward(state, 'exit', '')
                # n???u ch??a ph???i tr???ng th??i k???t th??c th?? t??m gt t???t nh???t l??m ph???n th?????ng
                else:
                    actions = self.mdp.getPossibleActions(state)
                    iterationValue[state] = max([self.computeQValueFromValues(state, action) for action in actions])
            self.values = iterationValue

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        temp=0
        for sprime, prob in self.mdp.getTransitionStatesAndProbs(state, action):
            temp += prob * (self.mdp.getReward(state, action, sprime) + self.discount * self.values[sprime])
        return  temp
        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        if self.mdp.isTerminal(state):
            return None
        actions = self.mdp.getPossibleActions(state)

        allActions = {}
        # t??m t???t c??? c??c h??nh ?????ng v?? gi?? tr??? t????ng ???ng 
        # r???i tr??? v??? h??nh ?????ng t????ng ???ng v???i gi?? tr??? l???n nh???t
        for action in actions:
            allActions[action] = self.computeQValueFromValues(state, action)
        
        return max(allActions, key=allActions.get)
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def computeQValues(self, state):
        # Tr??? v??? m???t couter ch???a t???t c??? qValues ??????t??? m???t tr???ng th??i nh???t ?????nh
        actions = self.mdp.getPossibleActions(state)  # All possible actions from a state
        qValues = util.Counter()  #  counter ch???a c???p (action, qValue) 

        for action in actions:
            # ????a Q value ???? t??nh cho action nh???t ?????nh v??o couter
            qValues[action] = self.computeQValueFromValues(state, action)

        return qValues

    def runValueIteration(self):

        for k in range(self.iterations):

            state = self.mdp.getStates()[k %  len(self.mdp.getStates())]
            best = self.computeActionFromValues(state)
            if best is None:
                V = 0
            else:
                V = self.computeQValueFromValues(state, best)
            self.values[state] = V


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        allStates = self.mdp.getStates()
        predecessors = dict()
        for state in allStates:
            predecessors[state]=set()
        for state in allStates:
            allactions=self.mdp.getPossibleActions(state)
            for a in allactions:
                possibleNextStates = self.mdp.getTransitionStatesAndProbs(state, a)
                for nextState,pred in possibleNextStates:
                    if pred>0:
                        predecessors[nextState].add(state)
        pq = util.PriorityQueue()
        for state in allStates:

            stateQValues = self.computeQValues(state)

            if len(stateQValues) > 0:
                maxQValue = stateQValues[stateQValues.argMax()]
                diff = abs(self.values[state] - maxQValue)
                pq.push(state, -diff)
        for i in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            stateQValues = self.computeQValues(state)
            maxQValue = stateQValues[stateQValues.argMax()]
            self.values[state] = maxQValue
            for p in predecessors[state]:

                pQValues = self.computeQValues(p)
                maxQValue = pQValues[pQValues.argMax()]
                diff = abs(self.values[p] - maxQValue)

                if diff > self.theta:
                    pq.update(p, -diff)

