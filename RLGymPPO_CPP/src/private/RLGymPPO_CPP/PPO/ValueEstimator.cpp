#include "ValueEstimator.h"

#include <torch/nn/modules/linear.h>
#include <torch/nn/modules/activation.h>
#include <torch/csrc/api/include/torch/serialize.h>
#include <fstream>

RLGPC::ValueEstimator::ValueEstimator(int inputAmount, const IList& layerSizes, torch::Device device, bool build_network) : device(device) {
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

		// Output layer, just gives 1 output for value estimate
		seq->push_back(nn::Linear(layerSizes.back(), 1));

		register_module("seq", seq);
	}

	this->to(device, true);
}

void RLGPC::ValueEstimator::Load(std::filesystem::path path, torch::Device device) {
	auto streamIn = std::ifstream(path, std::ios::binary);
	streamIn >> std::noskipws;
	if (!streamIn.good())
		throw std::runtime_error("File does not exist or can't be accessed.");
	torch::load(seq, streamIn, device);
}

void RLGPC::ValueEstimator::Save(std::filesystem::path path) const {
	auto streamOut = std::ofstream(path, std::ios::binary);
	torch::save(seq, streamOut);
}

std::vector<uint64_t> RLGPC::ValueEstimator::GetSizes() {
	std::vector<uint64_t> result = {};
	for (auto param : this->parameters()) {
		// See RLGPC::DiscretePolicy::GetSizes() for detailed explanation of why 
		// precise topological dimensions are used instead of `param.numel()`.
		result.push_back(param.dim());
		for (int64_t dim_size : param.sizes()) {
			result.push_back(dim_size);
		}
	}
	return result;
}