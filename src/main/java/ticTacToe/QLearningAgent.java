package ticTacToe;

import java.util.List;
import java.util.Random;

/**
 * A Q-Learning agent with a Q-Table, i.e. a table of Q-Values. This table is
 * implemented in the {@link QTable} class.
 * 
 * The methods to implement are:
 * (1) {@link QLearningAgent#train}
 * (2) {@link QLearningAgent#extractPolicy}
 * 
 * Your agent acts in a {@link TTTEnvironment} which provides the method
 * {@link TTTEnvironment#executeMove} which returns an {@link Outcome} object,
 * in other words
 * an [s,a,r,s']: source state, action taken, reward received, and the target
 * state after the opponent has played their move. You may want/need to edit
 * {@link TTTEnvironment} - but you probably won't need to.
 * 
 * @author ae187
 */

public class QLearningAgent extends Agent {

	/**
	 * The learning rate, between 0 and 1.
	 */
	double alpha = 0.5;

	/**
	 * The number of episodes to train for
	 */
	int numEpisodes = 10000;

	/**
	 * The discount factor (gamma)
	 */
	double discount = 0.9;

	/**
	 * The epsilon in the epsilon greedy policy used during training.
	 */
	double epsilon = 0.1;

	/**
	 * This is the Q-Table. To get an value for an (s,a) pair, i.e. a (game, move)
	 * pair.
	 * 
	 */

	QTable qTable = new QTable();

	/**
	 * This is the Reinforcement Learning environment that this agent will interact
	 * with when it is training.
	 * By default, the opponent is the random agent which should make your q
	 * learning agent learn the same policy
	 * as your value iteration and policy iteration agents.
	 */
	TTTEnvironment env = new TTTEnvironment();

	/**
	 * Construct a Q-Learning agent that learns from interactions with
	 * {@code opponent}.
	 * 
	 * @param opponent     the opponent agent that this Q-Learning agent will
	 *                     interact with to learn.
	 * @param learningRate This is the rate at which the agent learns. Alpha from
	 *                     your lectures.
	 * @param numEpisodes  The number of episodes (games) to train for
	 */
	public QLearningAgent(Agent opponent, double learningRate, int numEpisodes, double discount) {
		env = new TTTEnvironment(opponent);
		this.alpha = learningRate;
		this.numEpisodes = numEpisodes;
		this.discount = discount;
		initQTable();
		train();
	}

	/**
	 * Initialises all valid q-values -- Q(g,m) -- to 0.
	 * 
	 */

	protected void initQTable() {
		List<Game> allGames = Game.generateAllValidGames('X');// all valid games where it is X's turn, or it's terminal.
		for (Game g : allGames) {
			List<Move> moves = g.getPossibleMoves();
			for (Move m : moves) {
				this.qTable.addQValue(g, m, 0.0);
				// System.out.println("initing q value. Game:"+g);
				// System.out.println("Move:"+m);
			}

		}

	}

	/**
	 * Uses default parameters for the opponent (a RandomAgent) and the learning
	 * rate (0.2). Use other constructor to set these manually.
	 */
	public QLearningAgent() {
		this(new RandomAgent(), 0.1, 70000, 0.9);// training episodes set to 70000

	}

	/**
	 * Implement this method. It should play {@code this.numEpisodes} episodes of
	 * Tic-Tac-Toe with the TTTEnvironment, updating q-values according
	 * to the Q-Learning algorithm as required. The agent should play according to
	 * an epsilon-greedy policy where with the probability {@code epsilon} the
	 * agent explores, and with probability {@code 1-epsilon}, it exploits.
	 * 
	 * At the end of this method you should always call the {@code extractPolicy()}
	 * method to extract the policy from the learned q-values. This is currently
	 * done for you on the last line of the method.
	 */

	public void train() {

		// creating a random number greedy epsilon
		Random random = new Random();

		// looping through all episodes
		for (int episode = 0; episode < numEpisodes; episode++) {

			// resetting environment for new game
			env.reset();
			Game currentState = env.getCurrentGameState();

			// if the state is terminal add Q-value for this state and go to next episode
			if (currentState.isTerminal()) {
				qTable.addQValue(currentState, null, 0.0);
				continue;
			}

			// play till game ends(termianl state)
			while (!env.isTerminal()) {

				// var to store the selected move
				Move selectedMove;

				// epsilon greedy policy explore or exploit
				if (random.nextDouble() < epsilon) {

					// select a random move from available moves(explore)
					List<Move> possibleMoves = env.getPossibleMoves();
					selectedMove = possibleMoves.get(random.nextInt(possibleMoves.size()));
				} else {

					// selecting move with the highest Q-value (exploit)
					List<Move> possibleMoves = env.getPossibleMoves();
					selectedMove = possibleMoves.get(0);

					// looping through all possible moves and selecting one with the greatest
					// Q-value
					for (Move move : possibleMoves) {
						if (qTable.getQValue(currentState, move) > qTable.getQValue(currentState, selectedMove)) {

							// updating the best move if we find a move with greater Q-value
							selectedMove = move;
						}
					}
				}

				// executing the selected move and geting the outcome(next state, reward etc)
				Outcome outcome;
				try {
					// execute the move and get outcome
					outcome = env.executeMove(selectedMove);
				} catch (IllegalMoveException e) {

					// if illegal move give warning
					System.out.println("Illegal move encountered: " + e.getMessage());
					break;
				}

				// if outcome is null (terminal state) then break
				if (outcome == null) {
					break;
				}

				// calculate max Q-value for the next state (after opposite move)
				double maxNextQValue = 0.0;
				if (!outcome.sPrime.isTerminal()) {

					// if next state is not terminal we calculate the highest Q-value from all
					// possible moves
					for (Move move : outcome.sPrime.getPossibleMoves()) {
						maxNextQValue = Math.max(maxNextQValue, qTable.getQValue(outcome.sPrime, move));
					}
				}

				// updating Q-value for the current state using Q-learning foramula
				double sample = outcome.localReward + discount * maxNextQValue;

				double updatedQValue = (1 - alpha) * qTable.getQValue(outcome.s, selectedMove) + alpha * sample;
				qTable.addQValue(outcome.s, selectedMove, updatedQValue);

				// move to next state and update current stae to new state
				currentState = outcome.sPrime;
			}

			// decay epsilon on episodes for better exploitation
			epsilon = Math.max(0.1, epsilon * 0.995);
		}

		// --------------------------------------------------------
		// you shouldn't need to delete the following lines of code.
		this.policy = extractPolicy();
		if (this.policy == null) {
			System.out.println("Unimplemented methods! First implement the train() & extractPolicy methods");
			// System.exit(1);
		}
	}

	/**
	 * Implement this method. It should use the q-values in the {@code qTable} to
	 * extract a policy and return it.
	 *
	 * @return the policy currently inherent in the QTable
	 */

	public Policy extractPolicy() {

		// initializing new policy to store stae-action mapping
		Policy policy = new Policy();

		// generating all game states where it is Xs turn
		List<Game> allGames = Game.generateAllValidGames('X');

		// looping through each game state
		for (Game state : allGames) {

			// skipping terminal states
			if (state.isTerminal()) {
				continue;
			}

			// getting all possible moves for current state
			List<Move> possibleMoves = state.getPossibleMoves();

			// skipping if moves are not there
			if (possibleMoves.isEmpty()) {
				continue;
			}

			// vars to track best move with highest Q-value
			Move bestMove = null;
			double bestQValue = -10000;

			// iterating to find move with highest Q-value
			for (Move move : possibleMoves) {
				double qValue = qTable.getQValue(state, move);

				// if qValue greater than the best value we already have update best q value and
				// best move
				if (qValue > bestQValue) {
					bestQValue = qValue;
					bestMove = move;
				}
			}
			// if we find a best move add it to the policy map
			if (bestMove != null) {
				policy.policy.put(state, bestMove);
			}
		}
		// return the policy
		return policy;
	}

	public static void main(String a[]) throws IllegalMoveException {
		// Test method to play your agent against a human agent (yourself).
		QLearningAgent agent = new QLearningAgent();

		HumanAgent d = new HumanAgent();

		Game g = new Game(agent, d, d);
		g.playOut();

	}

}
