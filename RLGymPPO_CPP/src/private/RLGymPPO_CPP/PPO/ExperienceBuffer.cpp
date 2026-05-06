#include "ExperienceBuffer.h"

#include "../Util/TorchFuncs.h"

using namespace torch;

RLGPC::ExperienceBuffer::ExperienceBuffer(int64_t maxSize, int seed, torch::Device device) :
	maxSize(maxSize), seed(seed), device(device), rng(seed) {
	
}

void RLGPC::ExperienceBuffer::SubmitExperience(ExperienceTensors& _data) {
	RG_NOGRAD;

	bool empty = curSize == 0;

	for (size_t i = 0; i < ExperienceTensors::TENSOR_AMOUNT; i++) {
		Tensor& ourTen = data[i];
		Tensor& addTen = _data[i];

		int64_t addAmount = addTen.size(0);

		if (addAmount > maxSize) {
			addTen = addTen.slice(0, addAmount - maxSize);
			addAmount = maxSize;
		}

		if (empty) {
			// Initalize tensor
			auto sizes = addTen.sizes();
			auto newSizes = std::vector<int64_t>(sizes.begin(), sizes.end());
			newSizes[0] = maxSize;
			ourTen = torch::zeros(newSizes, device);
			ourTen.add_(NAN);
		}

		int64_t endSpace = maxSize - head;
		if (addAmount <= endSpace) {
			ourTen.slice(0, head, head + addAmount).copy_(addTen, true);
		} else {
			ourTen.slice(0, head, maxSize).copy_(addTen.slice(0, 0, endSpace), true);
			ourTen.slice(0, 0, addAmount - endSpace).copy_(addTen.slice(0, endSpace, addAmount), true);
		}
	}

	head = (head + _data[0].size(0)) % maxSize;
	curSize = RS_MIN(curSize + _data[0].size(0), maxSize);
}

RLGPC::ExperienceBuffer::SampleSet RLGPC::ExperienceBuffer::_GetSamples(const int64_t* indices, size_t size) const {
	
	auto tIndicesOptions = torch::TensorOptions().dtype(torch::kInt64);
	Tensor tIndices = torch::from_blob(const_cast<int64_t*>(indices), { (int64_t)size }, tIndicesOptions).clone().to(device, true, true);

	// TODO: Reptitive
	SampleSet result;
	result.actions = torch::index_select(data.actions, 0, tIndices);
	result.logProbs = torch::index_select(data.logProbs, 0, tIndices);
	result.states = torch::index_select(data.states, 0, tIndices);
	result.values = torch::index_select(data.values, 0, tIndices);
	result.advantages = torch::index_select(data.advantages, 0, tIndices);
	return result;
}

std::vector<RLGPC::ExperienceBuffer::SampleSet> RLGPC::ExperienceBuffer::GetAllBatchesShuffled(int64_t batchSize) {

	// Make list of shuffled sample indices
	int64_t* indices = new int64_t[curSize];
	std::iota(indices, indices + curSize, 0); // Fill ascending indices
	std::shuffle(indices, indices + curSize, rng);

	// Get a sample set from each of the batches
	std::vector<SampleSet> result;
	for (int64_t startIdx = 0; startIdx + batchSize <= curSize; startIdx += batchSize) {
		result.push_back(_GetSamples(indices + startIdx, batchSize));
	}

	delete[] indices;
	return result;
}

void RLGPC::ExperienceBuffer::Clear() {
	*this = ExperienceBuffer(maxSize, seed, device);
}
