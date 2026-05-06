#pragma once
#include <RLGymPPO_CPP/Lists.h>
#include "../FrameworkTorch.h"

namespace RLGPC {

	struct ExperienceTensors {
		torch::Tensor
			states, actions, logProbs, rewards, 

#ifdef RG_PARANOID_MODE
			debugCounters,
#endif

			nextStates, dones, truncated, values, advantages;

		constexpr static size_t TENSOR_AMOUNT =
#ifdef RG_PARANOID_MODE
			10;
#else
			9;
#endif

		torch::Tensor& operator[](size_t index) { 
#ifdef RG_PARANOID_MODE
            switch(index) {
                case 0: return states;
                case 1: return actions;
                case 2: return logProbs;
                case 3: return rewards;
                case 4: return debugCounters;
                case 5: return nextStates;
                case 6: return dones;
                case 7: return truncated;
                case 8: return values;
                case 9: return advantages;
                default: throw std::runtime_error("Invalid tensor index");
            }
#else
            switch(index) {
                case 0: return states;
                case 1: return actions;
                case 2: return logProbs;
                case 3: return rewards;
                case 4: return nextStates;
                case 5: return dones;
                case 6: return truncated;
                case 7: return values;
                case 8: return advantages;
                default: throw std::runtime_error("Invalid tensor index");
            }
#endif
        }
		const torch::Tensor& operator[](size_t index) const { 
			return const_cast<ExperienceTensors*>(this)->operator[](index);
		}
	};

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/ppo/experience_buffer.py
	class ExperienceBuffer {
	public:

		torch::Device device;
		int seed;

		ExperienceTensors data;

		int64_t curSize = 0;
		int64_t head = 0;
		int64_t maxSize;

		std::default_random_engine rng;

		ExperienceBuffer(int64_t maxSize, int seed, torch::Device device);

		void SubmitExperience(ExperienceTensors& data);

		struct SampleSet {
			torch::Tensor actions, logProbs, states, values, advantages;
		};
		SampleSet _GetSamples(const int64_t* indices, size_t size) const;

		// Not const because it uses our random engine
		std::vector<SampleSet> GetAllBatchesShuffled(int64_t batchSize);

		void Clear();

		


	};
}