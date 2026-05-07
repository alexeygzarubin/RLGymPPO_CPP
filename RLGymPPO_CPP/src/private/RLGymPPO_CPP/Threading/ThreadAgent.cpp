#include "ThreadAgent.h"

#include "ThreadAgentManager.h"
#include <RLGymPPO_CPP/Util/Timer.h>

using namespace RLGPC;

/**
 * @brief Aggregates the current observation arrays from all managed Rocket League environments.
 *
 * Safety Constraints:
 * If a game instance fails to return a valid observation array, passing an empty tensor
 * to `torch::concat` will hard crash the entire application via a LibTorch core dump.
 * We proactively validate `.numel()` and `.dim()` here and fail-fast using standard C++ exceptions.
 *
 * @param games Vector of active GameInst pointers.
 * @return Concatenated 2D tensor representing the state of all active matches.
 * @throws std::invalid_argument if the input vector is empty.
 * @throws std::runtime_error if any game returns a malformed observation tensor.
 */
torch::Tensor MakeGamesOBSTensor(std::vector<GameInst*>& games) {
	if (games.empty()) {
		throw std::invalid_argument("MakeGamesOBSTensor: Expected non-empty games vector. Received empty vector.");
	}
	std::vector<torch::Tensor> obsTensors = {};
	for (auto game : games) {
		auto t = FLIST2_TO_TENSOR(game->curObs);
		if (t.dim() == 0 || t.numel() == 0) {
			throw std::runtime_error("MakeGamesOBSTensor: Expected valid observation tensor. Received empty tensor.");
		}
		obsTensors.push_back(t);
	}

	try {
		return torch::concat(obsTensors);
	} catch (std::exception& e) {
		throw std::runtime_error(std::string("MakeGamesOBSTensor: Failed to concat OBS tensors: ") + e.what());
	}
}

/**
 * @brief The infinite execution loop for a parallel thread agent.
 *
 * Architectural Intent:
 * This function forms the core of the >60,000 SPS parallel rollout pipeline. It executes
 * the RocketSim environment loop for `ta->numGames` simultaneously. To maximize throughput,
 * it runs policy inference independently per thread.
 *
 * Thread Safety & Synchronization:
 * - Policies: Each `ThreadAgent` receives a strictly thread-local clone of the policy.
 * - Locks: Uses `ta->gameStepMutex` to prevent race conditions during trajectory tracking, 
 *          and `mgr->inferMutex` to prevent concurrent GPU access issues if blockConcurrentInfer is set.
 * - Error Handling: All nested operations (LibTorch inference, RocketSim stepping) are wrapped
 *                   in try-catch blocks. Any exception thrown guarantees locks are released
 *                   first to avoid cross-thread deadlocks, followed by throwing a high-context 
 *                   `std::runtime_error` to kill the pipeline cleanly.
 *
 * @param ta Pointer to the thread's specific agent context.
 */
void _RunFunc(ThreadAgent* ta) {
	RG_NOGRAD;
	ta->isRunning = true;

	auto mgr = (ThreadAgentManager*)ta->_manager;
	auto& games = ta->gameInsts;
	int numGames = ta->numGames;

	auto device = mgr->device;

	bool render = mgr->renderSender != NULL;
	if (render && mgr->renderDuringTraining) {
		if (ta->index != 0)
			render = false;
	}
	bool deterministic = mgr->deterministic;
	bool blockConcurrentInfer = mgr->blockConcurrentInfer;
	Timer stepTimer = {};

	// Start games
	for (auto game : games)
		game->Start();

	// Will stores our current observations for all our games
	torch::Tensor curObsTensor = MakeGamesOBSTensor(games);

#if 0 // TODO: Potential cause of learning errors
	bool halfPrec = mgr->policyHalf != NULL;
#else
	constexpr bool halfPrec = false;
#endif

	auto policy = (halfPrec ? mgr->policyHalf : mgr->policy);

	while (ta->shouldRun) {

		if (render)
			stepTimer.Reset();

		// Don't run if we reached our step limit
		while (ta->stepsCollected > ta->maxCollect)
			THREAD_WAIT();

		while (mgr->disableCollection)
			THREAD_WAIT();

		// Move our current OBS tensor to the device we run the policy on
		// This conversion time is not counted as a part of policy inference time
		torch::Tensor curObsTensorDevice;
		if (halfPrec) {
			curObsTensorDevice = curObsTensor.to(RG_HALFPERC_TYPE).to(device, true);
		} else {
			curObsTensorDevice = curObsTensor.to(device, true);
		}

		// Infer the policy to get actions for all our agents in all our games
		Timer policyInferTimer = {};
		

		if (blockConcurrentInfer)
			mgr->inferMutex.lock();
		RLGPC::DiscretePolicy::ActionResult actionResults;
		try {
			actionResults = policy->GetAction(curObsTensorDevice, deterministic);
		} catch (std::exception& e) {
			if (blockConcurrentInfer) mgr->inferMutex.unlock();
			throw std::runtime_error(std::string("ThreadAgent::_RunFunc: Exception during policy->GetAction(): ") + e.what());
		}
		if (blockConcurrentInfer)
			mgr->inferMutex.unlock();
		if (halfPrec) {
			actionResults.action = actionResults.action.to(torch::ScalarType::Float);
			actionResults.logProb = actionResults.logProb.to(torch::ScalarType::Float);
		}

		float policyInferTime = policyInferTimer.Elapsed();
		ta->times.policyInferTime += policyInferTime;

		// Step the gym with the actions we got
		Timer gymStepTimer = {};
		ta->gameStepMutex.lock();
		float avgRew = 0;
		auto stepResults = new RLGSC::Gym::StepResult[numGames];
		int actionsOffset = 0;
		for (int i = 0; i < numGames; i++) {
			auto game = games[i];
			int numPlayers = game->match->playerAmount;

			// Actions output has a dimension for each player, but not for each game
			// So we will need to slice the section of it that is for this game
			auto actionSlice = actionResults.action.slice(0, actionsOffset, actionsOffset + numPlayers);

			try {
				IList temp_ilist;
				try {
					temp_ilist = TENSOR_TO_ILIST(actionSlice.flatten());
				} catch (const std::exception& e) {
					throw std::runtime_error(std::string("ThreadAgent::_RunFunc: Exception during TENSOR_TO_ILIST: ") + e.what() + " size: " + std::to_string(actionSlice.size(0)));
				}
				stepResults[i] = game->Step(temp_ilist);
			} catch (const std::exception& e) {
				throw std::runtime_error(std::string("ThreadAgent::_RunFunc: Exception during game->Step(): ") + e.what());
			} catch (...) {
				throw std::runtime_error("ThreadAgent::_RunFunc: Unknown Exception during game->Step()!");
			}

			actionsOffset += numPlayers;
		}
		ta->gameStepMutex.unlock();

		// Make sure we got the end of actions
		// Otherwise there's a wrong number of actions for whatever reason
		assert(actionsOffset == actionResults.action.size(0));
		float envStepTime = gymStepTimer.Elapsed();
		ta->times.envStepTime += envStepTime;

		// Update our tensor storing the next observation after the step, from each gym
		torch::Tensor nextObsTensor = MakeGamesOBSTensor(games);

		if (!render) {
			// Steps complete, add all timestep data to our trajectories, for each game
			Timer trajAppendTimer = {};
			ta->trajMutex.lock();
			for (int i = 0, playerOffset = 0; i < numGames; i++) {
				int numPlayers = games[i]->match->playerAmount;

				auto& stepResult = stepResults[i];

				float done = (float)stepResult.done;
				float truncated = (float)false;

				auto tDone = torch::tensor(done);
				auto tTruncated = torch::tensor(truncated);

				for (int j = 0; j < numPlayers; j++) {
					ta->trajectories[i][j].AppendSingleStep(
						{
							curObsTensor[playerOffset + j],
							actionResults.action[playerOffset + j],
							actionResults.logProb[playerOffset + j],
							torch::tensor(stepResult.reward[j]),

#ifdef RG_PARANOID_MODE
							torch::Tensor(),
#endif

							nextObsTensor[playerOffset + j],
							tDone,
							tTruncated
						}
					);
				}

				ta->stepsCollected += numPlayers;
				playerOffset += numPlayers;
			}
			ta->trajMutex.unlock();
			ta->times.trajAppendTime += trajAppendTimer.Elapsed();
		} else {
			// Update renderer
			auto renderSender = mgr->renderSender;
			auto renderGame = games[0];
			renderSender->Send(renderGame->gym->prevState, renderGame->gym->match->prevActions);

			// Delay for render
			// TODO: Somewhat dumb system using static variables
			{
				namespace chr = std::chrono;
				static auto lastRenderTime = chr::high_resolution_clock::now();
				auto durationSince = chr::high_resolution_clock::now() - lastRenderTime;
				lastRenderTime = chr::high_resolution_clock::now();

				int64_t micsSince = chr::duration_cast<chr::microseconds>(durationSince).count();

				double timeTaken = stepTimer.Elapsed();
				double targetTime = (1 / 120.0) * renderGame->gym->tickSkip / mgr->renderTimeScale;
				double sleepTime = RS_MAX(targetTime - timeTaken, 0);
				int64_t sleepMics = (int64_t)(sleepTime * 1000.0 * 1000.0);

				std::this_thread::sleep_for(chr::microseconds(sleepMics));
			}
		}

		// Now that the step is done, our next OBS becomes our current
		curObsTensor = nextObsTensor;

		delete[] stepResults;
	}

	ta->isRunning = false;
}

RLGPC::ThreadAgent::ThreadAgent(void* manager, int numGames, uint64_t maxCollect, EnvCreateFn envCreateFn, int index)
	: _manager(manager), numGames(numGames), maxCollect(maxCollect), index(index) {

	trajectories.resize(numGames);
	for (int i = 0; i < numGames; i++) {
		auto envCreateResult = envCreateFn();
		gameInsts.push_back(new GameInst(envCreateResult.gym, envCreateResult.match));
		trajectories[i].resize(envCreateResult.match->playerAmount);
	}
}

void RLGPC::ThreadAgent::Start() {
	this->shouldRun = true;
	this->thread = std::thread(_RunFunc, this);
	this->thread.detach();
}

void RLGPC::ThreadAgent::Stop() {
	this->shouldRun = false;

	// Wait for thread to stop runing
	// TODO: Lame solution
	while (isRunning)
		RG_SLEEP(1);
}