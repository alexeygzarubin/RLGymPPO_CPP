#include "InferUnit.h"

#include <RLGymPPO_CPP/PPO/DiscretePolicy.h>
#include <RLGymPPO_CPP/PPO/ValueEstimator.h>
#include <RLGymPPO_CPP/FrameworkTorch.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <RLGymPPO_CPP/TransformerPolicy.h>

using namespace RLGSC;
using namespace RLGPC;

/**
 * @brief Constructs an inference unit for stand-alone evaluation or Python bridging.
 *
 * Safety Constraints:
 * `InferUnit` is typically used when binding the C++ models back to a Python environment
 * for visual evaluation or external processing. As such, it is entirely decoupled from 
 * the concurrent training architecture (`ThreadAgent` / `ThreadAgentManager`).
 * Checkpoint loading incorporates aggressive shape validation to prevent memory corruption
 * during out-of-process bindings.
 *
 * @param obsBuilder Observation extraction interface.
 * @param actionParser Interface converting tensor bounds into actionable controller inputs.
 * @param modelPath Disk path to the pre-trained LibTorch `.pt` module.
 * @param isPolicy Whether this unit evaluates an actor (Policy) or a critic (ValueEstimator).
 * @param obsSize Total length of the unrolled observation array.
 * @param layerSizes (Legacy) Layer sizes for generic initialization.
 * @param gpu Whether to force evaluation onto the GPU device.
 * @throws std::runtime_error if checkpoint deserialization fails.
 */
RLGPC::InferUnit::~InferUnit() = default;

RLGPC::InferUnit::InferUnit(
	OBSBuilder* obsBuilder, ActionParser* actionParser, 
	std::filesystem::path modelPath, bool isPolicy, int obsSize, const IList& layerSizes, bool gpu,
	std::string policy_type, int max_entities, int q_features, int kv_features, std::vector<int64_t> action_splits)
	: obsBuilder(obsBuilder), actionParser(actionParser) {

	RG_LOG("InferUnit():");

	if (!obsBuilder) throw std::invalid_argument("InferUnit::InferUnit: obsBuilder cannot be null.");
	if (!actionParser) throw std::invalid_argument("InferUnit::InferUnit: actionParser cannot be null.");

	RG_LOG(" > Creating policy/critic...");
	torch::Device device = gpu ? torch::kCUDA : torch::kCPU;

	if (policy_type == "EARL") {
		if (max_entities == 0 || q_features == 0 || kv_features == 0 || action_splits.empty()) {
			throw std::invalid_argument("InferUnit::InferUnit: policy_type is EARL but dimensions/splits are 0 or empty!");
		}
	}

	if (isPolicy) {
		if (policy_type == "EARL") {
			policy.reset(new TransformerPolicy(1, max_entities, q_features, kv_features, action_splits, device));
		} else {
			policy.reset(new DiscretePolicy(obsSize, actionParser->GetActionAmount(), layerSizes, device));
		}
		critic = nullptr;
	} else {
		policy = nullptr;
		if (policy_type == "EARL") {
			critic.reset(new TransformerValueEstimator(1, max_entities, q_features, kv_features, device));
		} else {
			critic.reset(new ValueEstimator(obsSize, layerSizes, device));
		}
	}
	RG_LOG(" > Loading policy/critic...");
	try {
		if (isPolicy) {
			policy->Load(modelPath, device);
		} else {
			critic->Load(modelPath, device);
		}
	} catch (std::exception& e) {
		throw std::runtime_error(std::string("InferUnit::InferUnit: Failed to load model, checkpoint may be corrupt or of different model arch. Exception: ") + e.what());
	}

	RG_LOG(" > Done!");
}

RLGSC::FList RLGPC::InferUnit::GetObs(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction) {
	return obsBuilder->BuildOBS(player, state, prevAction);
}

RLGSC::FList2 RLGPC::InferUnit::GetObs(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions) {
	FList2 obsSet = {};
	for (int i = 0; i < state.players.size(); i++)
		obsSet.push_back(obsBuilder->BuildOBS(state.players[i], state, prevActions[i]));
	return obsSet;
}

#define ASSERT_RIGHT_TYPE(name, otherName) \
if (name == NULL) RG_ERR_CLOSE("InferUnit: Failed to infer the " #name " because this inference unit was created to infer the " #otherName);

ActionSet RLGPC::InferUnit::InferPolicyAll(
	const GameState& state, const ActionSet& prevActions, 
	bool deterministic, float temperature
) {

	ASSERT_RIGHT_TYPE(policy, critic);

	std::lock_guard<std::mutex> lock(infer_mutex);

	FList2 obsSet = GetObs(state, prevActions);
	
	RG_NOGRAD;
	policy->temperature = temperature;
	torch::Tensor input_ten = FLIST2_TO_TENSOR(obsSet).to(policy->device);
	auto actionResult = policy->GetAction(input_ten, deterministic);
	auto actionParserInput = TENSOR_TO_ILIST(actionResult.action.flatten());

	return actionParser->ParseActions(actionParserInput, state);
}

Action RLGPC::InferUnit::InferPolicySingle(
	const PlayerData& player, const GameState& state, const Action& prevAction, 
	bool deterministic, float temperature
) {
	ASSERT_RIGHT_TYPE(policy, critic);

	std::lock_guard<std::mutex> lock(infer_mutex);

	FList obs = GetObs(player, state, prevAction);

	int playerIndex = 0;
	for (int i = 1; i < state.players.size(); i++) {
		if (state.players[i].carId == player.carId) {
			playerIndex = i;
			break;
		}
	}

	RG_NOGRAD;
	policy->temperature = temperature;
	torch::Tensor input_ten = torch::from_blob(obs.data(), {static_cast<long>(obs.size())}).to(policy->device);
	if (input_ten.dim() == 1) input_ten = input_ten.unsqueeze(0);
	auto actionResult = policy->GetAction(input_ten, deterministic);
	int act_amt = policy->actionAmount;
	IList actionParserInput = IList(state.players.size() * act_amt);
	auto flatAction = TENSOR_TO_ILIST(actionResult.action.flatten());
	for(int a=0; a < act_amt; ++a) {
		actionParserInput[playerIndex * act_amt + a] = flatAction[a];
	}

	return actionParser->ParseActions(actionParserInput, state)[playerIndex];
}

RLGSC::FList RLGPC::InferUnit::InferPolicySingleDistrib(
	const PlayerData& player, const GameState& state, const Action& prevAction,
	float temperature
) {
	ASSERT_RIGHT_TYPE(policy, critic);

	std::lock_guard<std::mutex> lock(infer_mutex);

	FList obs = GetObs(player, state, prevAction);

	RG_NOGRAD;
	policy->temperature = temperature;
	torch::Tensor input_ten = torch::from_blob(obs.data(), {static_cast<long>(obs.size())}).to(policy->device);
	if (input_ten.dim() == 1) input_ten = input_ten.unsqueeze(0);
	return TENSOR_TO_FLIST(policy->GetActionProbs(input_ten).reshape({ policy->actionAmount }));
}

RLGSC::FList RLGPC::InferUnit::InferCriticAll(const RLGSC::GameState& state, const RLGSC::ActionSet& prevActions) {
	ASSERT_RIGHT_TYPE(critic, policy);

	std::lock_guard<std::mutex> lock(infer_mutex);

	FList2 obsSet = GetObs(state, prevActions);

	RG_NOGRAD;
	torch::Tensor input_ten = FLIST2_TO_TENSOR(obsSet).to(critic->device);
	return TENSOR_TO_FLIST(critic->Forward(input_ten).cpu());
}

float RLGPC::InferUnit::InferCriticSingle(const RLGSC::PlayerData& player, const RLGSC::GameState& state, const RLGSC::Action& prevAction) {
	ASSERT_RIGHT_TYPE(critic, policy);

	std::lock_guard<std::mutex> lock(infer_mutex);

	FList obs = GetObs(player, state, prevAction);

	RG_NOGRAD;
	torch::Tensor input_ten = torch::from_blob(obs.data(), {static_cast<long>(obs.size())}).to(critic->device);
	if (input_ten.dim() == 1) input_ten = input_ten.unsqueeze(0);
	return critic->Forward(input_ten).cpu().item<float>();
}
