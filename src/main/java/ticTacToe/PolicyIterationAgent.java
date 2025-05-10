package ticTacToe;

import java.util.Date;
import java.util.HashMap;
import java.util.List;
import java.util.Random;

/**
 * A policy iteration agent. You should implement the following methods:
 * (1) {@link PolicyIterationAgent#evaluatePolicy}: this is the policy
 * evaluation step from your lectures
 * (2) {@link PolicyIterationAgent#improvePolicy}: this is the policy
 * improvement step from your lectures
 * (3) {@link PolicyIterationAgent#train}: this is a method that should
 * runs/alternate (1) and (2) until convergence.
 * 
 * NOTE: there are two types of convergence involved in Policy Iteration:
 * Convergence of the Values of the current policy,
 * and Convergence of the current policy to the optimal policy.
 * The former happens when the values of the current policy no longer improve by
 * much (i.e. the maximum improvement is less than
 * some small delta). The latter happens when the policy improvement step no
 * longer updates the policy, i.e. the current policy
 * is already optimal. The algorithm should stop when this happens.
 * 
 * @author ae187
 *
 */
public class PolicyIterationAgent extends Agent {

	/**
	 * This map is used to store the values of states according to the current
	 * policy (policy evaluation).
	 */
	HashMap<Game, Double> policyValues = new HashMap<Game, Double>();

	/**
	 * This stores the current policy as a map from {@link Game}s to {@link Move}.
	 */
	HashMap<Game, Move> curPolicy = new HashMap<Game, Move>();

	double discount = 0.9;

	/**
	 * The mdp model used, see {@link TTTMDP}
	 */
	TTTMDP mdp;

	/**
	 * loads the policy from file if one exists. Policies should be stored in .pol
	 * files directly under the project folder.
	 */
	public PolicyIterationAgent() {
		super();
		this.mdp = new TTTMDP();
		initValues();
		initRandomPolicy();
		train();

	}

	/**
	 * Use this constructor to initialise your agent with an existing policy
	 * 
	 * @param p
	 */
	public PolicyIterationAgent(Policy p) {
		super(p);

	}

	/**
	 * Use this constructor to initialise a learning agent with default MDP
	 * paramters (rewards, transitions, etc) as specified in
	 * {@link TTTMDP}
	 * 
	 * @param discountFactor
	 */
	public PolicyIterationAgent(double discountFactor) {

		this.discount = discountFactor;
		this.mdp = new TTTMDP();
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Use this constructor to set the various parameters of the Tic-Tac-Toe MDP
	 * 
	 * @param discountFactor
	 * @param winningReward
	 * @param losingReward
	 * @param livingReward
	 * @param drawReward
	 */
	public PolicyIterationAgent(double discountFactor, double winningReward, double losingReward, double livingReward,
			double drawReward) {
		this.discount = discountFactor;
		this.mdp = new TTTMDP(winningReward, losingReward, livingReward, drawReward);
		initValues();
		initRandomPolicy();
		train();
	}

	/**
	 * Initialises the {@link #policyValues} map, and sets the initial value of all
	 * states to 0
	 * (V0 under some policy pi ({@link #curPolicy} from the lectures). Uses
	 * {@link Game#inverseHash} and {@link Game#generateAllValidGames(char)} to do
	 * this.
	 * 
	 */
	public void initValues() {
		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames)
			this.policyValues.put(g, 0.0);

	}

	/**
	 * You should implement this method to initially generate a random policy, i.e.
	 * fill the {@link #curPolicy} for every state. Take care that the moves you
	 * choose
	 * for each state ARE VALID. You can use the {@link Game#getPossibleMoves()}
	 * method to get a list of valid moves and choose
	 * randomly between them.
	 */
	public void initRandomPolicy() {
		// creating a random object for selecting the randpm moves
		Random random = new Random();

		// getting all valid game states where it iss Xs turn or terminal state(end
		// state)
		List<Game> allGames = Game.generateAllValidGames('X');

		// looping through each game state
		for (Game state : allGames) {

			// skipping termianal state(moves are not needed for end state)
			if (state.isTerminal()) {
				continue;
			}
			// getting all possible moves for the current state
			List<Move> possibleMoves = state.getPossibleMoves();

			// randomly selecting one move from those possible moves
			Move randomMove = possibleMoves.get(random.nextInt(possibleMoves.size()));

			// mapping current gmae state to the randomly selected move in curPolicy
			curPolicy.put(state, randomMove);
		}
	}

	/**
	 * Performs policy evaluation steps until the maximum change in values is less
	 * than {@code delta}, in other words
	 * until the values under the currrent policy converge. After running this
	 * method,
	 * the {@link PolicyIterationAgent#policyValues} map should contain the values
	 * of each reachable state under the current policy.
	 * You should use the {@link TTTMDP} {@link PolicyIterationAgent#mdp} provided
	 * to do this.
	 *
	 * @param delta
	 */
	protected void evaluatePolicy(double delta) {
		// var to calculate maximum change in values
		double maximumChange;

		do {
			// setting max change to 0 at the start of every iteration
			maximumChange = 0;

			// looping through all game states in current policy
			for (Game state : curPolicy.keySet()) {

				// skipping termianl states(moves are not needed for eavluation)
				if (state == null || state.isTerminal()) {
					continue;
				}

				// geting current move
				Move currentMove = curPolicy.get(state);
				if (currentMove == null) {

					// skipping if the move is null
					System.out.println("Warning: Null move for state: " + state);
					continue;
				}
				// initializing expected value for this state to zero
				double expectedValue = 0.0;

				// generating possible trasitions for current state and current move
				List<TransitionProb> transitions = mdp.generateTransitions(state, currentMove);

				// if transitions are not there(null) give warning or skip if transitions are
				// there
				if (transitions == null || transitions.isEmpty()) {
					System.out.println("Warning: No transitions found for state: " + state + ", move: " + currentMove);
					continue;
				}
				// looping through transiitions for current state and current move
				for (TransitionProb transition : transitions) {

					// getting the next state
					Game nextState = transition.outcome.sPrime;

					// getting reward for the transition
					double reward = transition.outcome.localReward;

					// getting probability for the transition
					double probability = transition.prob;

					// calculating the value of this transition and adding it to expectedValue
					expectedValue += probability * (reward + discount * policyValues.get(nextState));
				}

				// calculating abs value for the new and old value
				double change = Math.abs(expectedValue - policyValues.get(state));

				// updating the expected value in policyValue map
				policyValues.put(state, expectedValue);

				// if the change is greater than maximum change then we update maximumChange
				// with greater one
				if (change > maximumChange) {
					maximumChange = change;
				}
			}

		}
		// keep doing till the maximum change value is less than delta
		while (maximumChange >= delta);

	}

	/**
	 * This method should be run AFTER the
	 * {@link PolicyIterationAgent#evaluatePolicy} train method to improve the
	 * current policy according to
	 * {@link PolicyIterationAgent#policyValues}. You will need to do a single step
	 * of expectimax from each game (state) key in
	 * {@link PolicyIterationAgent#curPolicy}
	 * to look for a move/action that potentially improves the current policy.
	 * 
	 * @return true if the policy improved. Returns false if there was no
	 *         improvement, i.e. the policy already returned the optimal actions.
	 */
	protected boolean improvePolicy() {
		boolean policyImproved = false;

		// looping through all the states in current policy
		for (Game state : curPolicy.keySet()) {

			// skipping terminal states
			if (state == null || state.isTerminal()) {
				continue;
			}

			// getting the current best move for this state
			Move currentBestMove = curPolicy.get(state);

			// skipping if the move is null
			if (currentBestMove == null) {
				System.out.println("Warning: Null current best move for state: " + state);
				continue;
			}

			// intializing best value to a low value to compare it
			double bestValue = -10000;

			// will store the best move for this state
			Move bestMove = null;

			// loopping through all possible moves for the current state
			for (Move move : state.getPossibleMoves()) {

				// skipping if move is null
				if (move == null) {
					System.out.println("Warning: Null move for state: " + state);
					continue;
				}
				// initializing the expected value for this move to zero
				double expectedValue = 0.0;

				// getting all transitions (next states, rewards, probalbilities) for this move
				List<TransitionProb> transitions = mdp.generateTransitions(state, move);

				// skippign if the transitions are null
				if (transitions == null || transitions.isEmpty()) {
					System.out.println("Warning: No transitions found for state: " + state + ", move: " + move);
					continue;
				}

				for (TransitionProb transition : transitions) {

					// getting the next state from this transition
					Game nextState = transition.outcome.sPrime;

					// getting reward and prob for this transition
					double reward = transition.outcome.localReward;
					double probability = transition.prob;

					// updating the expected value for this move based on reward, prob etc
					expectedValue += probability * (reward + discount * policyValues.get(nextState));
				}

				// if the expected value is greater than the best value we have update the best
				// value to this expected value and best move to this move
				if (expectedValue > bestValue) {
					bestValue = expectedValue;
					bestMove = move;
				}
			}
			// if the best move is different than the current best move in curPolicy then
			// update the policy to use the new best move
			if (bestMove != null && !bestMove.equals(currentBestMove)) {
				curPolicy.put(state, bestMove);

				/// to show policy is improved set it to true
				policyImproved = true;
			}
		}
		/// return the improved policy bool
		return policyImproved;
	}

	/**
	 * The (convergence) delta
	 */
	double delta = 0.1;

	/**
	 * This method should perform policy evaluation and policy improvement steps
	 * until convergence (i.e. until the policy
	 * no longer changes), and so uses your
	 * {@link PolicyIterationAgent#evaluatePolicy} and
	 * {@link PolicyIterationAgent#improvePolicy} methods.
	 */
	public void train() {
		// to check if policy changed during iterarion
		boolean policyChanged;

		do {
			// doing evaluatePolicy(policy evaluation)
			evaluatePolicy(delta);

			// doing improvePolicy(policy improvement)
			policyChanged = improvePolicy();

		}
		// repeat till the policy is not changed
		while (policyChanged);

		// converting the updated curPolicy map into a Policy object
		this.policy = new Policy(curPolicy);

	}

	public static void main(String[] args) throws IllegalMoveException {
		/**
		 * Test code to run the Policy Iteration Agent agains a Human Agent.
		 */
		PolicyIterationAgent pi = new PolicyIterationAgent();

		HumanAgent h = new HumanAgent();

		Game g = new Game(pi, h, h);

		g.playOut();

	}

}
