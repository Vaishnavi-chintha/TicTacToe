package ticTacToe;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A Value Iteration Agent, only very partially implemented. The methods to
 * implement are:
 * (1) {@link ValueIterationAgent#iterate}
 * (2) {@link ValueIterationAgent#extractPolicy}
 * 
 * You may also want/need to edit {@link ValueIterationAgent#train} - feel free
 * to do this, but you probably won't need to.
 * 
 * @author ae187
 *
 */
public class ValueIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states
	 */
	Map<Game, Double> valueFunction = new HashMap<Game, Double>();

	/**
	 * the discount factor
	 */
	double discount = 0.9;

	/**
	 * the MDP model
	 */
	TTTMDP mdp = new TTTMDP();

	/**
	 * the number of iterations to perform - feel free to change this/try out
	 * different numbers of iterations
	 */
	int k = 10;

	/**
	 * This constructor trains the agent offline first and sets its policy
	 */
	public ValueIterationAgent() {
		super();
		mdp = new TTTMDP();
		this.discount = 0.9;
		initValues();
		train();
	}

	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * 
	 * @param p
	 */
	public ValueIterationAgent(Policy p) {
		super(p);

	}

	public ValueIterationAgent(double discountFactor) {

		this.discount = discountFactor;
		mdp = new TTTMDP();
		initValues();
		train();
	}

	/**
	 * Initialises the {@link ValueIterationAgent#valueFunction} map, and sets the
	 * initial value of all states to 0
	 * (V0 from the lectures). Uses {@link Game#inverseHash} and
	 * {@link Game#generateAllValidGames(char)} to do this.
	 * 
	 */
	public void initValues() {

		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames)
			this.valueFunction.put(g, 0.0);
		// loops throught games when its X chance to play and adds game state and
		// 0(intial value) to value func(map)

	}

	public ValueIterationAgent(double discountFactor, double winReward, double loseReward, double livingReward,
			double drawReward) {// create agent with diff rewards
		this.discount = discountFactor;
		mdp = new TTTMDP(winReward, loseReward, livingReward, drawReward);
	}

	// helper functions:
	// helper function to calculate the expected value of a move
	// used in findBestMove(another helper function) and iterate

	private double calculateExpectedValue(Game state, Move move, Map<Game, Double> currentValues) {

		// intitalizimg the expected value to 0
		double expectedValue = 0.0;

		// Get all the possible transitions for the move from generateTransitions in
		// TTTMDP(gives poss states, rewards and probabilities)
		List<TransitionProb> transitions = mdp.generateTransitions(state, move);

		// loop through all the possible transitions
		for (TransitionProb transition : transitions) {

			// getting next staate
			Game nextState = transition.outcome.sPrime;

			// getting the reward
			double reward = transition.outcome.localReward;

			// getting probability of the transition
			double Prob = transition.prob;

			// calculating the expected value and adding it to expectedValue
			expectedValue += Prob * (reward + discount * currentValues.get(nextState));
		}
		// returning the calculated exoected value
		return expectedValue;
	}

	// helper function to find the best move from given state based on current value
	// used in extract policy
	// checks all possible moves adn returns one with highest expected value
	private Move findBestMove(Game state, Map<Game, Double> currentValues) {

		// initializing bestMove to null -- will store the move with highest value
		Move bestMove = null;

		// intializing maximumValue to -10000(very low value) so any move will be better
		double maximumValue = -10000;

		// Check each possible move in current state
		for (Move move : state.getPossibleMoves()) {

			// calculate expected value of current move using the first helper function
			double moveValue = calculateExpectedValue(state, move, currentValues);

			// if the expected value of the move is greater than current max value update
			// best move to this move and max value to the move value
			if (moveValue > maximumValue) {
				maximumValue = moveValue;
				bestMove = move;
			}
		}
		// return the best move with highest expected value
		return bestMove;
	}

	/**
	 * 
	 * 
	 * /*
	 * Performs {@link #k} value iteration steps. After running this method, the
	 * {@link ValueIterationAgent#valueFunction} map should contain
	 * the (current) values of each reachable state. You should use the
	 * {@link TTTMDP} provided to do this.
	 * 
	 *
	 */
	public void iterate() {
		// loop for k number of iterations
		for (int iteration = 0; iteration < k; iteration++) {

			// creating a temporary map to store those updated state values during the
			// iteration
			Map<Game, Double> updatedStateValues = new HashMap<>();

			// Loop through each game state in the current value function
			for (Game currentState : valueFunction.keySet()) {

				// if the current state is terminal (that is end of the game) the value will not
				// change
				// we copy the value from valueFunction to updatedStateValues and continue
				if (currentState.isTerminal()) {
					updatedStateValues.put(currentState, valueFunction.get(currentState));
					continue;
				}

				// initializing maximumValue to -10000(very low value) so any move will be
				// better
				double maximumValue = -10000;

				// for each possible move in the current state
				for (Move move : currentState.getPossibleMoves()) {

					// we call calculateExpectedValue to get the expected value of that move
					double moveValue = calculateExpectedValue(currentState, move, valueFunction);

					// updating maximum value if this moves value is greater than the current
					// maximum we have
					maximumValue = Math.max(maximumValue, moveValue);
				}
				// we store the highest value ( that is the best moves expected value) for this
				// state in the updated map
				updatedStateValues.put(currentState, maximumValue);
			}
			// after all the states are dome we update valueFunction with those new values
			valueFunction = updatedStateValues;
		}
	}

	/**
	 * This method should be run AFTER the train method to extract a policy
	 * according to {@link ValueIterationAgent#valueFunction}
	 * You will need to do a single step of expectimax from each game (state) key in
	 * {@link ValueIterationAgent#valueFunction}
	 * to extract a policy.
	 * 
	 * @return the policy according to {@link ValueIterationAgent#valueFunction}
	 */
	public Policy extractPolicy() {
		// initializing a new policy object to storee the mapping of states to their
		// best moves
		Policy policy = new Policy();

		// looping through each state in value function
		for (Game state : valueFunction.keySet()) {

			// skippinf terminal state becuase end-state doesnt need any more moves(game is
			// already finished)
			if (state.isTerminal()) {
				continue;
			}
			// use findBestMove helper function to find the best move for current state
			Move bestMove = findBestMove(state, valueFunction);

			// if the hepler finds a best move i.e not null
			if (bestMove != null) {

				// best move is added to the new policy
				policy.policy.put(state, bestMove);
			}
		}
		// returning the policy with best moves for each state
		return policy;
	}

	/**
	 * This method solves the mdp using your implementation of
	 * {@link ValueIterationAgent#extractPolicy} and
	 * {@link ValueIterationAgent#iterate}.
	 */
	public void train() {
		/**
		 * First run value iteration
		 */
		this.iterate();
		/**
		 * now extract policy from the values in
		 * {@link ValueIterationAgent#valueFunction} and set the agent's policy
		 * 
		 */

		super.policy = extractPolicy();

		if (this.policy == null) {
			System.out.println("Unimplemented methods! First implement the iterate() & extractPolicy() methods");
			// System.exit(1);
		}

	}

	public static void main(String a[]) throws IllegalMoveException {
		// Test method to play the agent against a human agent.
		ValueIterationAgent agent = new ValueIterationAgent();
		HumanAgent d = new HumanAgent();

		Game g = new Game(agent, d, d);
		g.playOut();

	}
}
