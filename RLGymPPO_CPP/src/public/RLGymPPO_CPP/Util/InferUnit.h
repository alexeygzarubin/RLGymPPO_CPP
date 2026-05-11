#pragma once
#include "../Lists.h"
#include "../Threading/GameInst.h"
#include "../LearnerConfig.h"

#include <memory>
#include <mutex>

namespace RLGPC {
	class RG_IMEXPORT InferUnit {
	public:

		RLGSC::OBSBuilder* obsBuilder;
		RLGSC::ActionParser* actionParser;
		std::unique_ptr<class DiscretePolicy> policy;
		std::unique_ptr<class ValueEstimator> critic;
		std::mutex infer_mutex;

		~InferUnit();

		InferUnit(
			RLGSC::OBSBuilder* obsBuilder, RLGSC::ActionParser* actionParser, 
			std::filesystem::path modelPath, bool isPolicy, int obsSize, const RLGPC::IList& layerSizes, bool gpu = false,
			std::string policy_type = "MLP", int max_entities = 0, int q_features = 0, int kv_features = 0, std::vector<int64_t> action_splits = {});

		RLGSC::FList GetObs(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction);
		RLGSC::FList2 GetObs(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions);

		RLGSC::ActionSet InferPolicyAll(
			const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions, 
			bool deterministic, float temperature = 1.0f
		);
		RLGSC::Action InferPolicySingle(
			const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction, 
			bool deterministic, float temperature = 1.0f
		);
		RLGSC::FList InferPolicySingleDistrib(
			const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction, 
			float temperature = 1.0f
		);

		RLGSC::FList InferCriticAll(
			const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions
		);
		float InferCriticSingle(
			const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction
		);
	};
}