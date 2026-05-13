#include "DiscretePolicy.h"

#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/activation.h>
#include <private/RLGymPPO_CPP/FrameworkTorch.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <fstream>

RLGPC::DiscretePolicy::DiscretePolicy(int inputAmount, int actionAmount, const IList& layerSizes, torch::Device device, float temperature, bool build_network) :
	device(device), inputAmount(inputAmount), actionAmount(actionAmount), layerSizes(layerSizes), temperature(temperature) {
	using namespace torch;

	if (build_network) {
		seq = {};
		seq->push_back(nn::Linear(inputAmount, layerSizes[0]));
		seq->push_back(nn::ReLU());

		int prevLayerSize = layerSizes[0];
		for (int i = 1; i < layerSizes.size(); i++) {
			int layerSize = layerSizes[i];
			seq->push_back(nn::Linear(prevLayerSize, layerSize));
			seq->push_back(nn::ReLU());
			prevLayerSize = layerSize;
		}

		// Output layer, for each action
		seq->push_back(nn::Linear(layerSizes.back(), actionAmount));

		register_module("seq", seq);
	}

	this->to(device, true);
}

RLGPC::DiscretePolicy* RLGPC::DiscretePolicy::Clone() const {
	return new DiscretePolicy(inputAmount, actionAmount, layerSizes, device, temperature);
}

std::vector<uint64_t> RLGPC::DiscretePolicy::GetSizes() const {
	std::vector<uint64_t> result = {};
	for (auto param : this->parameters())
		result.push_back(param.numel());
	return result;
}

void RLGPC::DiscretePolicy::Save(std::filesystem::path path) const {
	auto streamOut = std::ofstream(path, std::ios::binary);
	torch::save(seq, streamOut);
}

void RLGPC::DiscretePolicy::Load(std::filesystem::path path, torch::Device device) {
	auto streamIn = std::ifstream(path, std::ios::binary);
	streamIn >> std::noskipws;
	if (!streamIn.good())
		throw std::runtime_error("File does not exist or can't be accessed.");
	torch::load(seq, streamIn, device);
}

void RLGPC::DiscretePolicy::CopyTo(DiscretePolicy& to) {
	RG_NOGRAD;
	try {
		auto fromParams = this->parameters();
		auto toParams = to.parameters();
		for (int i = 0; i < fromParams.size(); i++) {
			toParams[i].copy_(fromParams[i], false);
		}
	} catch (std::exception& e) {
		RG_ERR_CLOSE("DiscretePolicy::CopyTo() exception: " << e.what());
	}
}

torch::Tensor RLGPC::DiscretePolicy::GetOutput(torch::Tensor input) {
	auto baseOutput = seq->forward(input) / temperature;
	auto result = torch::nn::functional::softmax(
		baseOutput,
		torch::nn::functional::SoftmaxFuncOptions(-1)
	);

	if (actionProbBonuses.defined()) {
		result = result + actionProbBonuses.view({ 1, -1 });
		result = result / result.sum(-1).view({ -1, 1 });
	}

	return result;
}

torch::Tensor RLGPC::DiscretePolicy::GetActionProbs(torch::Tensor obs) {
	auto probs = GetOutput(obs);
	probs = probs.view({ -1, actionAmount });
	probs = torch::clamp(probs, ACTION_MIN_PROB, 1);
	return probs;
}

RLGPC::DiscretePolicy::ActionResult RLGPC::DiscretePolicy::GetAction(torch::Tensor obs, bool deterministic) {
	auto probs = GetActionProbs(obs);

	if (deterministic) {
		auto action = probs.argmax(1);
		return { action.cpu().flatten(), torch::zeros(action.numel()) };
	} else {
		auto action = torch::multinomial(probs, 1, true);
		auto logProb = torch::log(probs).gather(-1, action);
		return ActionResult{ action.cpu().flatten(), logProb.cpu().flatten() };
	}
}

RLGPC::DiscretePolicy::BackpropResult RLGPC::DiscretePolicy::GetBackpropData(torch::Tensor obs, torch::Tensor acts) {
	// Get probability of each action
	acts = acts.to(torch::kInt64, true);
	auto probs = GetActionProbs(obs);

	// Compute log probs and entropy
	auto logProbs = torch::log(probs);
	auto actionLogProbs = logProbs.gather(-1, acts);
	auto entropy = -(logProbs * probs);

	if (actionEntropyScales.defined()) {
		entropy = entropy * actionEntropyScales.view({ 1, -1 });
	}

	entropy = entropy.sum(-1);

	return BackpropResult{ actionLogProbs.to(device, true), entropy.to(device).mean() };
}
