#pragma once
#include <RLGymPPO_CPP/PPO/DiscretePolicy.h>
#include <RLGymPPO_CPP/EARLPerceiver.h>
#include <RLGymPPO_CPP/ControlsPredictorDiscrete.h>

#include <RLGymPPO_CPP/PPO/ValueEstimator.h>

namespace RLGPC {

/**
 * @brief Utility for parsing flattened 1D observations into 3D tensors for the Perceiver.
 */
struct RG_IMEXPORT EARLObservationParser {
    struct ParsedObs {
        torch::Tensor q;
        torch::Tensor kv;
        torch::Tensor mask;
    };

    static ParsedObs Unflatten(torch::Tensor input, int num_q, int num_entities, int query_features, int kv_features);
};

/**
 * @brief A LibTorch module wrapping the EARLPerceiver Transformer for RL policy inference.
 *
 * This class inherits from `DiscretePolicy` to integrate seamlessly into the 
 * RLGymPPO_CPP engine. It performs the crucial task of mapping flattened
 * ExperienceBuffer state vectors into the [Batch, Entities, Features] shapes
 * required by the Transformer.
 * 
 * Memory Management:
 * The policy heavily relies on RAII and polymorphic object lifecycles. All torch::nn 
 * sub-modules are stored as `nullptr` until registered during construction.
 * 
 * Thread Safety Constraints:
 * 16 parallel RocketSim instances rely on this class. Instances of this policy must be 
 * strictly isolated to their originating thread using the `Clone()` pattern. 
 * Cross-thread state sharing (such as concurrently modifying gradients or device contexts)
 * will result in race conditions and severe memory access violations.
 */
class RG_IMEXPORT TransformerPolicy : public DiscretePolicy {
public:
    EARLPerceiver perceiver_{nullptr};
    ControlsPredictorDiscrete predictor_{nullptr};

    int num_q_;
    int num_entities_;
    int query_features_;
    int kv_features_;
    std::vector<int64_t> splits_;

    TransformerPolicy(
        int num_q, int num_entities, int query_features, int kv_features,
        std::vector<int64_t> splits,
        torch::Device device, float temperature = 1.0f);

    DiscretePolicy* Clone() const override;

    void Save(std::filesystem::path path) const override;
    void Load(std::filesystem::path path, torch::Device device) override;

    void CopyTo(DiscretePolicy& to) override;

    // Overrides
    torch::Tensor GetOutput(torch::Tensor input) override;
    torch::Tensor GetActionProbs(torch::Tensor obs) override;
    ActionResult GetAction(torch::Tensor obs, bool deterministic) override;
    BackpropResult GetBackpropData(torch::Tensor obs, torch::Tensor acts) override;

    ~TransformerPolicy() override = default;
};

} // namespace RLGPC

namespace RLGPC {

/**
 * @brief A LibTorch module wrapping the EARLPerceiver Transformer for RL value estimation.
 */
class RG_IMEXPORT TransformerValueEstimator : public ValueEstimator {
public:
    EARLPerceiver perceiver_{nullptr};
    torch::nn::Sequential head_{nullptr};

    int num_q_;
    int num_entities_;
    int query_features_;
    int kv_features_;

    TransformerValueEstimator(
        int num_q, int num_entities, int query_features, int kv_features,
        torch::Device device);

    torch::Tensor Forward(torch::Tensor input) override;
    void Load(std::filesystem::path path, torch::Device device) override;
    void Save(std::filesystem::path path) const override;
    std::vector<uint64_t> GetSizes() override;
};

} // namespace RLGPC
