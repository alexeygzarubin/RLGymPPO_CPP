#include "PPOLearner.h"
#include <RLGymPPO_CPP/TransformerPolicy.h>

#include "../Util/TorchFuncs.h"

#include <torch/nn/utils/convert_parameters.h>
#include <torch/nn/utils/clip_grad.h>
#include <torch/csrc/api/include/torch/serialize.h>

using namespace torch;

Tensor _CopyParams(nn::Module* mod) {
	return torch::nn::utils::parameters_to_vector(mod->parameters()).cpu();
}

void _CopyModelParamsHalf(nn::Module* from, nn::Module* to) {
	RG_NOGRAD;
	try {
		auto fromParams = from->parameters();
		auto toParams = to->parameters();
		for (int i = 0; i < fromParams.size(); i++) {
			auto scaledParams = fromParams[i].to(RG_HALFPERC_TYPE);
			toParams[i].copy_(scaledParams, false);
		}
	} catch (std::exception& e) {
		throw std::runtime_error(std::string("PPOLearner::_CopyModelParamsHalf() exception: ") + e.what());
	}
}

RLGPC::PPOLearner::PPOLearner(int obsSpaceSize, int actSpaceSize, PPOLearnerConfig _config, Device _device) 
	: config(_config), device(_device) {

	if (config.miniBatchSize == 0)
		config.miniBatchSize = config.batchSize;

	if (config.batchSize % config.miniBatchSize != 0)
		RG_ERR_CLOSE("PPOLearner: config.batchSize must be a multiple of config.miniBatchSize");

	if (config.policy_type == "EARL") {
		policy = std::make_unique<TransformerPolicy>(1, config.max_entities, config.q_features, config.kv_features, config.action_splits, device, config.policyTemperature);
		valueNet = std::make_unique<TransformerValueEstimator>(1, config.max_entities, config.q_features, config.kv_features, device);
	} else {
		policy = std::make_unique<DiscretePolicy>(obsSpaceSize, actSpaceSize, config.policyLayerSizes, device, config.policyTemperature);
		valueNet = std::make_unique<ValueEstimator>(obsSpaceSize, config.criticLayerSizes, device);
	}

	if (config.halfPrecModels) {
		policyHalf = std::unique_ptr<DiscretePolicy>(policy->Clone());
		if (config.policy_type == "EARL") {
			valueNetHalf = std::make_unique<TransformerValueEstimator>(1, config.max_entities, config.q_features, config.kv_features, device);
		} else {
			valueNetHalf = std::make_unique<ValueEstimator>(obsSpaceSize, config.criticLayerSizes, device);
		}

		_CopyModelParamsHalf(policy.get(), policyHalf.get());
		_CopyModelParamsHalf(valueNet.get(), valueNetHalf.get());

		policyHalf->to(RG_HALFPERC_TYPE);
		valueNetHalf->to(RG_HALFPERC_TYPE);
	} else {
		policyHalf = nullptr;
		valueNetHalf = nullptr;
	}

	if (config.policy_type == "EARL") {
		policyOptimizer = std::make_unique<optim::AdamW>(policy->parameters(), optim::AdamWOptions(config.policyLR).weight_decay(config.weightDecay));
		valueOptimizer = std::make_unique<optim::AdamW>(valueNet->parameters(), optim::AdamWOptions(config.criticLR).weight_decay(config.weightDecay));
	} else {
		policyOptimizer = std::make_unique<optim::Adam>(policy->parameters(), optim::AdamOptions(config.policyLR));
		valueOptimizer = std::make_unique<optim::Adam>(valueNet->parameters(), optim::AdamOptions(config.criticLR));
	}
	valueLossFn = nn::MSELoss();

	if (config.measureGradientNoise) {
		noiseTrackerPolicy = std::make_unique<GradNoiseTracker>(config.batchSize, config.gradientNoiseUpdateInterval, config.gradientNoiseAvgDecay);
		noiseTrackerValueNet = std::make_unique<GradNoiseTracker>(config.batchSize, config.gradientNoiseUpdateInterval, config.gradientNoiseAvgDecay);
	} else {
		noiseTrackerPolicy = nullptr;
		noiseTrackerValueNet = nullptr;
	}
}

void RLGPC::PPOLearner::Learn(ExperienceBuffer* expBuffer, Report& report) {
	
	bool autocast = config.autocastLearn;

	if (autocast) {
#ifndef RG_CUDA_SUPPORT
		RG_ERR_CLOSE("Autocast not supported on non-CUDA!")
#endif
	}

	static std::unique_ptr<amp::GradScaler> gradScaler = nullptr;
#ifdef RG_CUDA_SUPPORT
	if (autocast && !gradScaler) {
		RG_LOG("Creating grad scaler...");
		gradScaler = std::make_unique<amp::GradScaler>();
	}
#endif

	int
		numIterations = 0,
		numMinibatchIterations = 0;
	float
		meanEntropy = 0,
		meanDivergence = 0,
		meanValLoss = 0,
		meanRatio = 0;
	FList clipFractions = {};

	// Save parameters first
	auto policyBefore = _CopyParams(policy.get());
	auto criticBefore = _CopyParams(valueNet.get());

	bool trainPolicy = config.policyLR != 0;
	bool trainCritic = config.criticLR != 0;

	Timer totalTimer = {};
	for (int epoch = 0; epoch < config.epochs; epoch++) {

		// Get randomly-ordered timesteps for PPO
		auto batches = expBuffer->GetAllBatchesShuffled(config.batchSize);

		for (auto& batch : batches) {
			auto batchActs = batch.actions;
			auto batchOldProbs = batch.logProbs;
			auto batchObs = batch.states;
			auto batchTargetValues = batch.values;
			auto batchAdvantages = batch.advantages;

			batchActs = batchActs.view({ config.batchSize, -1 });
			policyOptimizer->zero_grad();
			valueOptimizer->zero_grad();

			std::mutex threadUpdateMutex;

			auto fnRunMinibatch = [&](int start, int stop) {

				auto recordTime = [&](const std::string& name, double elapsed) {
					std::lock_guard<std::mutex> lock(threadUpdateMutex);
					report.Accum(name, elapsed);
				};

				float batchSizeRatio = (stop - start) / (float)config.batchSize;

				// Send everything to the device and enforce correct shapes
				auto acts = batchActs.slice(0, start, stop).to(device, true, true);
				auto obs = batchObs.slice(0, start, stop).to(device, true, true);
				
				auto advantages = batchAdvantages.slice(0, start, stop).to(device, true, true);
				auto oldProbs = batchOldProbs.slice(0, start, stop).to(device, true, true);
				auto targetValues = batchTargetValues.slice(0, start, stop).to(device, true, true);

				Timer timer = {};
				if (autocast) RG_AUTOCAST_ON();
				auto vals = valueNet->Forward(obs); // 11%
				recordTime("PPO Value Estimate Time", timer.Elapsed());

				timer.Reset();
				torch::Tensor logProbs, entropy, ratio, clipped, policyLoss, ppoLoss;
				if (trainPolicy) {
					// Get policy log probs & entropy
					DiscretePolicy::BackpropResult bpResult = policy->GetBackpropData(obs, acts); // 13%

					logProbs = bpResult.actionLogProbs;
					entropy = bpResult.entropy;

					logProbs = logProbs.view_as(oldProbs);
					recordTime("PPO Backprop Data Time", timer.Elapsed());

					// Compute PPO loss
					ratio = exp(logProbs - oldProbs);
					{
						std::lock_guard<std::mutex> lock(threadUpdateMutex);
						meanRatio += ratio.mean().detach().cpu().item<float>();
					}
					clipped = clamp(
						ratio, 1 - config.clipRange, 1 + config.clipRange
					);

					// Compute policy loss
					policyLoss = -min(
						ratio * advantages, clipped * advantages
					).mean();
					ppoLoss = (policyLoss - entropy * config.entCoef) * batchSizeRatio;
				}

				torch::Tensor valueLoss;
				if (trainCritic) {
					// Compute value loss
					vals = vals.view_as(targetValues);
					valueLoss = valueLossFn(vals, targetValues) * batchSizeRatio;
				}

				if (autocast) RG_AUTOCAST_OFF();

				float kl;
				if (trainPolicy) {
					// Compute KL divergence & clip fraction using SB3 method for reporting
					float clipFraction;
					{
						RG_NOGRAD;

						auto logRatio = logProbs - oldProbs;
						auto klTensor = (exp(logRatio) - 1) - logRatio;
						kl = klTensor.mean().detach().cpu().item<float>();

						clipFraction = mean((abs(ratio - 1) > config.clipRange).to(kFloat)).cpu().item<float>();
						{
							std::lock_guard<std::mutex> lock(threadUpdateMutex);
							clipFractions.push_back(clipFraction);
						}
					}
				}

				//timer.Reset();
				// NOTE: These gradient calls are a substantial portion of learn time
				//	From my testing, they are around 61% of learn time
				//	Results will probably vary heavily depending on model size and GPU strength
				if (autocast) {
					if (trainPolicy)
						gradScaler->scale(ppoLoss).backward();
					if (trainCritic)
						gradScaler->scale(valueLoss).backward();
				} else {
					try {
						if (trainPolicy)
							ppoLoss.backward(); // 29%
						if (trainCritic)
							valueLoss.backward(); // 24%
					} catch (const std::exception& e) {
						std::cout << "CRITICAL ERROR: PyTorch backward pass failed: " << e.what() << std::endl;
						throw;
					}
				}

				recordTime("PPO Gradient Time", timer.Elapsed());
				{
					std::lock_guard<std::mutex> lock(threadUpdateMutex);

					if (trainCritic)
						meanValLoss += valueLoss.cpu().detach().item<float>();
					if (trainPolicy) {
						meanDivergence += kl;
						meanEntropy += entropy.cpu().detach().item<float>();
					}
					numMinibatchIterations += 1;
				}


			};

			if (this->device.is_cpu()) {

				if (!this->minibatchThreadPool) {
					int numThreads = std::thread::hardware_concurrency();
					numThreads += numThreads / 2; // Seems to be slightly faster
					this->minibatchThreadPool = new ThreadPool(numThreads);
				}

				// Use multithreaded PPO learn
				int realMinibatchSize = config.batchSize / this->minibatchThreadPool->threads.size();

				for (int mbs = 0; mbs < config.batchSize; mbs += realMinibatchSize) {
					int start = mbs;
					int stop = start + realMinibatchSize;
					stop = RS_MIN(stop, config.batchSize);

					this->minibatchThreadPool->StartJob(std::bind(fnRunMinibatch, start, stop));
				}

				this->minibatchThreadPool->WaitAll();

			} else {
				for (int mbs = 0; mbs < config.batchSize; mbs += config.miniBatchSize) {
					int start = mbs;
					int stop = start + config.miniBatchSize;
					fnRunMinibatch(start, stop);
				}
			}

			if (config.measureGradientNoise) {
				if (trainPolicy)
					noiseTrackerPolicy->Update(policy->seq);
				if (trainCritic)
					noiseTrackerValueNet->Update(valueNet->seq);
			}

			if (trainPolicy)
				nn::utils::clip_grad_norm_(policy->parameters(), 0.5f);
			if (trainCritic)
				nn::utils::clip_grad_norm_(valueNet->parameters(), 0.5f);
			

			if (autocast) {
				if (trainPolicy)
					gradScaler->step(*policyOptimizer);
				if (trainCritic)
					gradScaler->step(*valueOptimizer);
			} else {
				if (trainPolicy)
					policyOptimizer->step();
				if (trainCritic)
					valueOptimizer->step();
			}

			if (policyHalf)
				_CopyModelParamsHalf(policy.get(), policyHalf.get());
			if (valueNetHalf)
				_CopyModelParamsHalf(valueNet.get(), valueNetHalf.get());
			
			if (autocast)
				gradScaler->update();
			numIterations += 1;
		}
	}

	numIterations = RS_MAX(numIterations, 1);
	numMinibatchIterations = RS_MAX(numMinibatchIterations, 1);

	// Compute averages for the metrics that will be reported
	meanEntropy /= numMinibatchIterations;
	meanDivergence /= numMinibatchIterations;
	meanValLoss /= numMinibatchIterations;
	meanRatio /= numMinibatchIterations;

	float meanClip = 0;
	if (!clipFractions.empty()) {
		for (float f : clipFractions)
			meanClip += f;
		meanClip /= clipFractions.size();
	}

	// Compute magnitude of updates made to the policy and value estimator
	auto policyAfter = _CopyParams(policy.get());
	auto criticAfter = _CopyParams(valueNet.get());

	float policyUpdateMagnitude = (policyBefore - policyAfter).norm().item<float>();
	float criticUpdateMagnitude = (criticBefore - criticAfter).norm().item<float>();

	float totalTime = totalTimer.Elapsed();

	// Assemble and return report
	cumulativeModelUpdates += numIterations;
	report["PPO Batch Consumption Time"] = totalTime / numIterations;
	report["Cumulative Model Updates"] = cumulativeModelUpdates;
	report["Policy Entropy"] = meanEntropy;
	report["Mean KL Divergence"] = meanDivergence;
	report["Mean Ratio"] = meanRatio;
	report["Value Function Loss"] = meanValLoss;
	report["SB3 Clip Fraction"] = meanClip;
	report["Policy Update Magnitude"] = policyUpdateMagnitude;
	report["Value Function Update Magnitude"] = criticUpdateMagnitude;
	report["PPO Learn Time"] = totalTimer.Elapsed();

	if (config.measureGradientNoise) {
		if (noiseTrackerPolicy->lastNoiseScale != 0)
			report["Grad Noise Policy"] = noiseTrackerPolicy->lastNoiseScale;
		if (noiseTrackerValueNet->lastNoiseScale != 0)
			report["Grad Noise Value Net"] = noiseTrackerValueNet->lastNoiseScale;
	}

	policyOptimizer->zero_grad();
	valueOptimizer->zero_grad();
}

// Get sizes of all parameters in a sequence
std::vector<uint64_t> GetSeqSizes(torch::nn::Sequential& seq) {
	std::vector<uint64_t> result = {};

	for (int i = 0; i < seq->size(); i++)
		for (auto param : seq[i]->parameters())
			result.push_back(param.numel());

	return result;
}

constexpr const char* MODEL_FILE_NAMES[] = {
		"PPO_POLICY.lt",
		"PPO_CRITIC.lt",
};

constexpr const char* OPTIM_FILE_NAMES[] = {
	"PPO_POLICY_OPTIM.lt",
	"PPO_CRITIC_OPTIM.lt",
};

void TorchLoadSaveSeq(torch::nn::Sequential seq, std::filesystem::path path, c10::Device device, bool load) {
	if (load) {
		auto streamIn = std::ifstream(path, std::ios::binary);
		streamIn >> std::noskipws;

		if (!streamIn.good())
			RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

		auto sizesBefore = GetSeqSizes(seq);

		try {
			torch::load(seq, streamIn, device);
		} catch (std::exception& e) {
			throw std::runtime_error(std::string("TorchLoadSaveSeq: Failed to load model, checkpoint may be corrupt or of different model arch. Exception: ") + e.what());
		}

		auto sizesAfter = GetSeqSizes(seq);
		if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {
			std::stringstream stream;
			stream << "TorchLoadSaveSeq: Saved model has different size than current model, cannot load model from " << path << ":\n";
			
			for (int i = 0; i < 2; i++) {
				stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
				for (uint64_t size : (i ? sizesAfter : sizesBefore))
					stream << size << ' ';

				stream << " ]";
				if (i == 0)
					stream << ",\n";
			}

			throw std::runtime_error(stream.str());
		}

	} else {
		auto streamOut = std::ofstream(path, std::ios::binary);
		torch::save(seq, streamOut);
	}
}

/**
 * @brief Safely serializes or deserializes a polymorphic DiscretePolicy module.
 *
 * Safety Constraints:
 * LibTorch will happily load a `.pt` model checkpoint of a completely different
 * architecture or shape size into memory without throwing an error natively. This causes 
 * catastrophic out-of-bounds segfaults during the first forward pass.
 * To mitigate this, this function caches the parameter topological sizes before loading,
 * and aggressively compares them post-load. If a shape mismatch is detected, it fails-fast
 * and aborts rather than executing with a corrupted computational graph.
 *
 * @param policy Pointer to the instantiated DiscretePolicy module.
 * @param path File system path to the model checkpoint.
 * @param device Target tensor device (CPU/CUDA).
 * @param load True to load from disk, false to save to disk.
 * @throws std::runtime_error if shape mismatch occurs or the filesystem is inaccessible.
 */
void TorchLoadSaveModule(RLGPC::DiscretePolicy* policy, std::filesystem::path path, c10::Device device, bool load) {
	if (load) {
		if (!std::filesystem::exists(path))
			RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

		auto sizesBefore = policy->GetSizes();

		try {
			policy->Load(path, device);
		} catch (std::exception& e) {
			throw std::runtime_error(std::string("TorchLoadSaveModule: Failed to load model, checkpoint may be corrupt or of different model arch. Exception: ") + e.what());
		}

		// Torch will happily load in a model of a totally different size, then we will crash when we try to use it
		// So we need to manually check if it is the same size
		auto sizesAfter = policy->GetSizes();
		if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {
			std::stringstream stream;
			stream << "Saved model has different size than current model, cannot load model from " << path << ":\n";
			
			for (int i = 0; i < 2; i++) {
				stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
				for (uint64_t size : (i ? sizesAfter : sizesBefore))
					stream << size << ' ';

				stream << " ]";
				if (i == 0)
					stream << ",\n";
			}

			throw std::runtime_error(stream.str());
		}

	} else {
		policy->Save(path);
	}
}

void TorchLoadSaveValueEstimator(RLGPC::ValueEstimator* valueNet, std::filesystem::path path, c10::Device device, bool load) {
	if (load) {
		if (!std::filesystem::exists(path))
			RG_ERR_CLOSE("Failed to load from " << path << ", file does not exist or can't be accessed");

		auto sizesBefore = valueNet->GetSizes();

		try {
			valueNet->Load(path, device);
		} catch (std::exception& e) {
			throw std::runtime_error(std::string("TorchLoadSaveValueEstimator: Failed to load model, checkpoint may be corrupt or of different model arch. Exception: ") + e.what());
		}

		auto sizesAfter = valueNet->GetSizes();
		if (!std::equal(sizesBefore.begin(), sizesBefore.end(), sizesAfter.begin(), sizesAfter.end())) {
			std::stringstream stream;
			stream << "Saved value estimator has different size than current model, cannot load model from " << path << ":\n";
			
			for (int i = 0; i < 2; i++) {
				stream << " > " << (i ? "Saved model:   [ " : "Current model: [ ");
				for (uint64_t size : (i ? sizesAfter : sizesBefore))
					stream << size << ' ';

				stream << " ]";
				if (i == 0)
					stream << ",\n";
			}

			throw std::runtime_error(stream.str());
		}

	} else {
		valueNet->Save(path);
	}
}

/**
 * @brief Manages the safe disk serialization of all PPO components (Policy, ValueNet, Optimizers).
 *
 * Architectural Intent:
 * Ensures atomic state recovery for the entire PPO pipeline. Handles synchronization 
 * of Half-Precision (FP16) copies if configured, ensuring the mixed-precision 
 * inference graph matches the newly loaded FP32 master weights.
 *
 * @param learner Pointer to the active PPO learner instance.
 * @param folderPath Directory containing the training checkpoints.
 * @param load True to load states, false to save states.
 * @throws std::runtime_error on I/O failures or corrupted checkpoint topology.
 */
void TorchLoadSaveAll(RLGPC::PPOLearner* learner, std::filesystem::path folderPath, bool load) {

	if (load) {
		for (int i = 0; i < 2; i++) {
			if (!std::filesystem::exists(folderPath / MODEL_FILE_NAMES[i]))
				throw std::runtime_error("TorchLoadSaveAll: PPOLearner failed to find required model file \"" + std::string(MODEL_FILE_NAMES[i]) + "\" in " + folderPath.string() + ".");
		}
	}

	TorchLoadSaveModule(learner->policy.get(), folderPath / MODEL_FILE_NAMES[0], learner->device, load);
	TorchLoadSaveValueEstimator(learner->valueNet.get(), folderPath / MODEL_FILE_NAMES[1], learner->device, load);

	if (load) {
		if (learner->policyHalf)
			_CopyModelParamsHalf(learner->policy.get(), learner->policyHalf.get());
		if (learner->valueNetHalf)
			_CopyModelParamsHalf(learner->valueNet.get(), learner->valueNetHalf.get());
	}

	// Load or save optimizers
	if (load) {
		try {
			for (int i = 0; i < 2; i++) {
				auto path = folderPath / OPTIM_FILE_NAMES[i];

				if (!std::filesystem::exists(path)) {
					throw std::runtime_error("TorchLoadSaveAll: Missing required optimizer file at " + path.string());
				}

				{ // Check if empty
					std::ifstream testStream = std::ifstream(path, std::istream::ate | std::ios::binary);
					if (testStream.tellg() == 0) {
						throw std::runtime_error("TorchLoadSaveAll: Saved optimizer file is completely empty (0 bytes) at " + path.string());
					}
				}

				DataStreamIn in = DataStreamIn(path, false);

				auto& optim = i ? learner->valueOptimizer : learner->policyOptimizer;

				torch::serialize::InputArchive policyOptArchive;
				policyOptArchive.load_from(path.string(), learner->device);
				(i ? learner->valueOptimizer : learner->policyOptimizer)->load(policyOptArchive);
			}

		} catch (std::exception& e) {
			throw std::runtime_error(std::string("TorchLoadSaveAll: Failed to load optimizers, exception: ") + e.what() + "\nCheckpoint may be corrupt.");
		}
	} else {
		for (int i = 0; i < 2; i++) {
			torch::serialize::OutputArchive policyOptArchive;
			(i ? learner->valueOptimizer : learner->policyOptimizer)->save(policyOptArchive);
			policyOptArchive.save_to((folderPath / OPTIM_FILE_NAMES[i]).string());
		}
	}
}

void RLGPC::PPOLearner::SaveTo(std::filesystem::path folderPath) {
	RG_LOG("PPOLearner(): Saving models to: " << folderPath);
	TorchLoadSaveAll(this, folderPath, false);
}

RLGPC::DiscretePolicy* RLGPC::PPOLearner::LoadAdditionalPolicy(std::filesystem::path folderPath) {
	std::filesystem::path policyPath = folderPath / MODEL_FILE_NAMES[0];
	if (!std::filesystem::exists(policyPath))
		return NULL;

	RLGPC::DiscretePolicy* newPolicy = policy->Clone();
	TorchLoadSaveModule(newPolicy, policyPath, newPolicy->device, true);
	return newPolicy;
}

void RLGPC::PPOLearner::LoadFrom(std::filesystem::path folderPath)  {
	RG_LOG("PPOLearner(): Loading models from: " << folderPath);
	if (!std::filesystem::is_directory(folderPath))
		RG_ERR_CLOSE("PPOLearner:LoadFrom(): Path " << folderPath << " is not a valid directory");

	TorchLoadSaveAll(this, folderPath, true);

	// Explicitly move to device after loading (fixes silent CPU fallback bug)
	policy->to(device);
	valueNet->to(device);

	UpdateLearningRates(config.policyLR, config.criticLR);
}

void RLGPC::PPOLearner::UpdateLearningRates(float policyLR, float criticLR) {
	config.policyLR = policyLR;
	config.criticLR = criticLR;

	if (config.policy_type == "EARL") {
		for (auto& g : policyOptimizer->param_groups())
			static_cast<torch::optim::AdamWOptions&>(g.options()).lr(policyLR);

		for (auto& g : valueOptimizer->param_groups())
			static_cast<torch::optim::AdamWOptions&>(g.options()).lr(criticLR);
	} else {
		for (auto& g : policyOptimizer->param_groups())
			static_cast<torch::optim::AdamOptions&>(g.options()).lr(policyLR);

		for (auto& g : valueOptimizer->param_groups())
			static_cast<torch::optim::AdamOptions&>(g.options()).lr(criticLR);
	}

	std::stringstream updatedMsg;
	updatedMsg << std::scientific << "Updated learning rate to [" << policyLR << ", " << criticLR << "]";
	RG_LOG("PPOLearner: " << updatedMsg.str());
}