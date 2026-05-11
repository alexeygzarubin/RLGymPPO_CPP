#pragma once
#include <RLGymPPO_CPP/Lists.h>

#include <torch/nn/modules/container/sequential.h>

namespace RLGPC {
	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/value_estimator.py
	class ValueEstimator : public torch::nn::Module {
	public:
		torch::Device device;
		torch::nn::Sequential seq;

		ValueEstimator(int inputAmount, const IList& layerSizes, torch::Device device, bool build_network = true);
		virtual ~ValueEstimator() = default;

		virtual torch::Tensor Forward(torch::Tensor input) {
			return seq->forward(input).to(device, true);
		}
		virtual void Load(std::filesystem::path path, torch::Device device);
		virtual void Save(std::filesystem::path path) const;
		virtual std::vector<uint64_t> GetSizes();
	};
}