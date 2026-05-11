#pragma once
#include <RLGymPPO_CPP/Lists.h>

#include <torch/nn/modules/container/sequential.h>
#include <torch/nn/functional.h>

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/discrete_policy.py
	class RG_IMEXPORT DiscretePolicy : public torch::nn::Module {
	public:
		torch::Device device;
		torch::nn::Sequential seq;
		int inputAmount;
		int actionAmount;
		IList layerSizes;

		float temperature;
		torch::Tensor actionEntropyScales;
		torch::Tensor actionProbBonuses;

		// Min probability that an action will be taken
		constexpr static float ACTION_MIN_PROB = 1e-11;

		DiscretePolicy(int inputAmount, int actionAmount, const IList& layerSizes, torch::Device device, float temperature = 1, bool build_network = true);

		RG_NO_COPY(DiscretePolicy);

		virtual DiscretePolicy* Clone() const;

		virtual std::vector<uint64_t> GetSizes() const;
		virtual void Save(std::filesystem::path path) const;
		virtual void Load(std::filesystem::path path, torch::Device device);

		virtual void CopyTo(DiscretePolicy& to);

		virtual torch::Tensor GetOutput(torch::Tensor input);

		virtual torch::Tensor GetActionProbs(torch::Tensor obs);

		struct ActionResult {
			torch::Tensor action, logProb;
		};
		virtual ActionResult GetAction(torch::Tensor obs, bool deterministic);
		
		struct BackpropResult {
			torch::Tensor actionLogProbs;
			torch::Tensor entropy;
		};
		virtual BackpropResult GetBackpropData(torch::Tensor obs, torch::Tensor acts);

		~DiscretePolicy() = default;
	};
}