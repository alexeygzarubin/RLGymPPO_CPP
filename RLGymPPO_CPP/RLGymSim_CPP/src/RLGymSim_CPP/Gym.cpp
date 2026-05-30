#include "Gym.h"

namespace RLGSC {

	template<int PlayerData::* DATA_VAR>
	void IncPlayerCounter(Car* car, void* userInfo) {
		if (!car)
			return;

		Gym* gym = (Gym*)userInfo;
		for (auto& player : gym->prevState.players)
			if (player.carId == car->id)
				(player.*DATA_VAR)++;
	}

	void _ShotEventCallback(Arena* arena, Car* shooter, Car* passer, void* userInfo) {
		IncPlayerCounter<&PlayerData::matchShots>(shooter, userInfo);
		IncPlayerCounter<&PlayerData::matchShotPasses>(passer, userInfo);
	}

	void _GoalEventCallback(Arena* arena, Car* scorer, Car* passer, void* userInfo) {
		IncPlayerCounter<&PlayerData::matchGoals>(scorer, userInfo);
		IncPlayerCounter<&PlayerData::matchAssists>(passer, userInfo);
	}

	void _SaveEventCallback(Arena* arena, Car* saver, void* userInfo) {
		IncPlayerCounter<&PlayerData::matchSaves>(saver, userInfo);
	}

	void _BumpCallback(Arena* arena, Car* bumper, Car* victim, bool isDemo, void* userInfo) {
		if (bumper->team == victim->team)
			return;

		IncPlayerCounter<&PlayerData::matchBumps>(bumper, userInfo);

		if (isDemo)
			IncPlayerCounter<&PlayerData::matchDemos>(bumper, userInfo);
	}

	Gym::Gym(Match* match, int tickSkip, CarConfig carConfig, GameMode gameMode, MutatorConfig mutatorConfig) :
		match(match), tickSkip(tickSkip), 
		// yexela-c: Fixes severe event-tracking bug by setting actionDelay to 0 instead of tickSkip - 1 (7).
		// By doing this, we step the environment for the full 8 ticks in a single Step(8) call.
		// All event callbacks (bumps, demos, saves, shots, assists, goals) will trigger and successfully
		// propagate their counters into state/prevState, rather than being wiped out by the prevState copy.
		actionDelay(0) {
		arena = Arena::Create(gameMode);
		arena->SetMutatorConfig(mutatorConfig);

		for (int i = 0; i < match->teamSize; i++) {
			arena->AddCar(Team::BLUE, carConfig);
		}
		if (match->spawnOpponents) {
			for (int i = 0; i < match->teamSize; i++) {
				arena->AddCar(Team::ORANGE, carConfig);
			}
		}

		eventTracker.SetShotCallback(_ShotEventCallback, this);
		eventTracker.SetGoalCallback(_GoalEventCallback, this);
		eventTracker.SetSaveCallback(_SaveEventCallback, this);

		arena->SetCarBumpCallback(_BumpCallback, this);
	}

	FList2 Gym::Reset() {
		GameState resetState = match->ResetState(arena);
		match->EpisodeReset(resetState);
		prevState = resetState;
		eventTracker.ResetPersistentInfo();

		FList2 obs = match->BuildObservations(resetState);
		return obs;
	}

	Gym::StepResult Gym::Step(const ActionParser::Input& actionsData) {
		try {
			// std::cout << "Gym::Step: Parsing actions" << std::endl;
			ActionSet actions = match->ParseActions(actionsData, prevState);
			match->prevActions = actions;

			GameState state;

			{ // Step arena with actions
				// std::cout << "Gym::Step: Stepping arena" << std::endl;
				for (int i = 0; i < actions.size(); i++) {
					Car* car = arena->GetCar(prevState.players[i].carId);
					if (!car) {
						throw std::runtime_error("Gym::Step FAILED: Car ID " + std::to_string(prevState.players[i].carId) + " found in prevState but missing from Arena!");
					}
					car->controls = (CarControls)actions[i];
				}

				arena->Step(tickSkip - actionDelay);
				if (arena->gameMode != GameMode::HEATSEEKER)
					eventTracker.Update(arena);
				state = prevState; // All callbacks have been hit
				state.UpdateFromArena(arena);
				arena->Step(actionDelay);
				totalTicks += tickSkip;
				totalSteps++;
			}

			// std::cout << "Gym::Step: Building observations" << std::endl;
			FList2 obs = match->BuildObservations(state);
			
			// std::cout << "Gym::Step: Checking IsDone" << std::endl;
			bool done = match->IsDone(state);
			
			// std::cout << "Gym::Step: Getting Rewards" << std::endl;
			FList rewards = match->GetRewards(state, done);
			prevState = state;

			return StepResult {
				obs,
				rewards,
				done,
				state
			};
		} catch (const std::exception& e) {
			std::cout << "CRITICAL ERROR: Gym::Step FAILED with: " << e.what() << std::endl;
			std::exit(1);
		}
	}
}