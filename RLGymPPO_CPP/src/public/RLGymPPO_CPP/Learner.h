#pragma once

#include "Threading/GameInst.h"
#include "Util/WelfordRunningStat.h"
#include "Util/MetricSender.h"
#include "Util/RenderSender.h"
#include "LearnerConfig.h"
#include <memory>
#include <functional>
#include <filesystem>

namespace torch {
    namespace nn {
        class Module;
    }
}

namespace RLGPC {

	typedef std::function<void(class Learner*, Report&)> IterationCallback;
	typedef std::function<void(class Learner*, std::filesystem::path)> SaveCallback;

	// https://github.com/AechPro/rlgym-ppo/blob/main/rlgym_ppo/learner.py
	class RG_IMEXPORT Learner {
	public:
		const bool inferDeviceIsCPU;
		LearnerConfig config;

		std::unique_ptr<class PPOLearner> ppo;
		std::unique_ptr<class ThreadAgentManager> agentMgr;
		std::unique_ptr<class ExperienceBuffer> expBuffer;
		EnvCreateFn envCreateFn;
		std::unique_ptr<MetricSender> metricSender;
		std::unique_ptr<RenderSender> renderSender;
		std::unique_ptr<class DiscretePolicy> policyInfer;
		std::unique_ptr<class DiscretePolicy> policyInferHalf;
		class DiscretePolicy* GetTrainingPolicy();
		torch::nn::Module* GetTrainingPolicyModule();
		void* globalGilRelease = NULL;

		std::unique_ptr<struct SkillTracker> skillTracker;

		int obsSize;
		int actionAmount;

		std::string runID = {};

		uint64_t
			totalTimesteps = 0,
			totalEpochs = 0;
			
		WelfordRunningStat returnStats = WelfordRunningStat(1);

		Learner(EnvCreateFn envCreateFunc, LearnerConfig config);
		void Learn();
		void AddNewExperience(class GameTrajectory& gameTraj, Report& report);

		void UpdateLearningRates(float policyLR, float criticLR);
        void SetEntropyCoef(float newCoef);

        void SetActionProbBonuses(RLGSC::FList newVals);
        void SetActionEntropyScales(RLGSC::FList newVals);

		std::vector<Report> GetAllGameMetrics();

		void Save();
		void Load();
		void SaveStats(std::filesystem::path path);
		void LoadStats(std::filesystem::path path);

		IterationCallback iterationCallback = NULL;
		StepCallback stepCallback = NULL;
		SaveCallback saveCallback = nullptr;

		RG_NO_COPY(Learner);

		~Learner();
	};
}