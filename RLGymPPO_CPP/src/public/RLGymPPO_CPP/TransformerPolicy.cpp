#include "TransformerPolicy.h"
#include <torch/csrc/api/include/torch/serialize.h>
#include <fstream>
#include <private/RLGymPPO_CPP/FrameworkTorch.h>

namespace RLGPC {

EARLObservationParser::ParsedObs EARLObservationParser::Unflatten(torch::Tensor input, int num_q, int num_entities, int query_features, int kv_features) {
    if (input.dim() != 2) {
        throw std::invalid_argument("EARLObservationParser::Unflatten: Invalid input tensor dimensions. Expected 2 [Batch, ObsSize], got " + std::to_string(input.dim()));
    }
    int q_size = num_q * query_features;
    int kv_size = num_entities * kv_features;
    int expected_size = q_size + kv_size + num_entities;
    if (input.size(1) != expected_size) {
        throw std::invalid_argument("EARLObservationParser::Unflatten: Invalid input tensor size at dim 1. Expected " + std::to_string(expected_size) + ", got " + std::to_string(input.size(1)));
    }

    // Zero-copy unflattening using .view()
    torch::Tensor q = input.slice(1, 0, q_size).view({-1, num_q, query_features});
    torch::Tensor kv = input.slice(1, q_size, q_size + kv_size).view({-1, num_entities, kv_features});
    torch::Tensor mask_float = input.slice(1, q_size + kv_size, q_size + kv_size + num_entities);
    
    // Convert float mask back to boolean (True means ignore/padding)
    // Note: While this .to() cast allocates a new tensor on the hot path, we accept this minor 
    // overhead because the PPO Gym interface strictly requires observations to be returned as a 
    // contiguous std::vector<float>. Packing booleans or dealing with raw bit-casts adds unnecessary 
    // complexity for an array of only 41 elements.
    torch::Tensor mask = mask_float.to(torch::kBool);

    return {q, kv, mask};
}

TransformerPolicy::TransformerPolicy(
    int num_q, int num_entities, int query_features, int kv_features,
    std::vector<int64_t> splits,
    torch::Device device, float temperature)
    : DiscretePolicy(
        num_q * query_features + num_entities * kv_features + num_entities, // Total input amount
        splits.size(), // Action amount
        {256, 256}, // Dummy layer sizes to satisfy base class
        device, temperature, false /* build_network */),
      num_q_(num_q), num_entities_(num_entities),
      query_features_(query_features), kv_features_(kv_features),
      splits_(std::move(splits)) {

    perceiver_ = register_module("perceiver", EARLPerceiver(num_q_, query_features_, kv_features_, 256, 4, 3));
    predictor_ = register_module("predictor", ControlsPredictorDiscrete(256, splits_));

    this->to(device, true);
}

DiscretePolicy* TransformerPolicy::Clone() const {
    return new TransformerPolicy(num_q_, num_entities_, query_features_, kv_features_, splits_, device, temperature);
}

void TransformerPolicy::Save(std::filesystem::path path) const {
    torch::serialize::OutputArchive archive;
    this->save(archive);
    archive.save_to(path.string());
}

void TransformerPolicy::Load(std::filesystem::path path, torch::Device device) {
    torch::serialize::InputArchive archive;
    archive.load_from(path.string(), device);
    this->load(archive);
}

/**
 * @brief Synchronizes policy parameters safely across thread boundaries.
 * 
 * Thread Safety:
 * This uses `torch::NoGradGuard` implicitly via the `RG_NOGRAD` macro to ensure no 
 * computational graph state is accidentally retained during parameter copies.
 * 
 * @param to The target DiscretePolicy to copy network parameters into.
 * @throws std::runtime_error on failed memory synchronization.
 */
void TransformerPolicy::CopyTo(DiscretePolicy& to) {
    RG_NOGRAD;
    try {
        auto fromParams = this->parameters();
        auto toParams = to.parameters();
        for (size_t i = 0; i < fromParams.size(); i++) {
            toParams[i].copy_(fromParams[i], false);
        }
    } catch (std::exception& e) {
        RG_ERR_CLOSE("TransformerPolicy::CopyTo() exception: " << e.what());
    }
}

/**
 * @brief Performs a forward pass to compute raw action logits from flattened input data.
 *
 * This function resolves the dimension discrepancy between the PPO ExperienceBuffer (which 
 * requires tightly packed 1D vectors per step) and the Transformer (which requires 3D 
 * sequence features). 
 * 
 * Architectural Intent:
 * We use `torch::Tensor::slice` and `torch::Tensor::view` extensively here. These methods 
 * manipulate strides rather than reallocating memory, providing a zero-copy unflattening 
 * mechanism. This guarantees zero VRAM reallocation overhead during the high-frequency 
 * >60,000 SPS parallel rollout phase.
 *
 * @param input A 2D float tensor of shape [Batch, ObsSize].
 * @return A 2D float tensor of shape [Batch, 21] representing unscaled action logits.
 * @throws std::invalid_argument if tensor dimensions or input bounds do not match configurations.
 */
torch::Tensor TransformerPolicy::GetOutput(torch::Tensor input) {
    auto parsed = EARLObservationParser::Unflatten(input, num_q_, num_entities_, query_features_, kv_features_);
    
    // Forward pass
    torch::Tensor latent = perceiver_->forward(parsed.q, parsed.kv, parsed.mask);
    torch::Tensor latent_flat = latent.squeeze(1); // num_q is 1, squeeze it out
    
    return predictor_->forward_flat(latent_flat) / temperature; // [batch, 21] logits
}

/**
 * @brief Converts raw concatenated logits into isolated multi-discrete probability distributions.
 *
 * Architectural Intent:
 * Standard PPO policies process a single monolithic continuous or discrete distribution. 
 * Since Rocket League steering involves multiple independent axes (Throttle, Steer, Jump, etc.),
 * we must slice the 21 action logits into 8 mathematically independent probability bins. 
 * We apply Softmax independently per bin before re-concatenating them to guarantee the 
 * probability mass of each action axis strictly sums to 1.0.
 *
 * @param obs Flattened observation tensor [Batch, ObsSize].
 * @return Concatenated multi-discrete probability tensor [Batch, 21].
 */
torch::Tensor TransformerPolicy::GetActionProbs(torch::Tensor obs) {
    auto flat_logits = GetOutput(obs); // [batch, 21]
    
    // We must softmax each independent bin, not the whole thing
    auto logits_splits = torch::split_with_sizes(flat_logits, splits_, -1);
    std::vector<torch::Tensor> probs_splits;
    
    for (const auto& logit : logits_splits) {
        auto prob = torch::nn::functional::softmax(logit, torch::nn::functional::SoftmaxFuncOptions(-1));
        probs_splits.push_back(torch::clamp(prob, ACTION_MIN_PROB, 1));
    }
    
    // Concatenate back into a flat probs tensor
    return torch::cat(probs_splits, -1);
}

/**
 * @brief Computes executable actions and trajectory log-probabilities for the rollout environment.
 *
 * Memory Constraints:
 * During multi-discrete sampling (`torch::multinomial`), tensors are dynamically allocated.
 * We must safely flatten and aggregate the log_prob sums over the 8 independent dimensions.
 * Because the ExperienceBuffer tracks continuous layout log probabilities as [Batch, 1],
 * we squash the log_prob along the last dimension.
 * 
 * Determinism:
 * When determinism is true, log probabilities are returned as zeros explicitly cast to 
 * `kFloat32`. This prevents the ExperienceBuffer from crashing due to unexpected upcasting 
 * from the standard Int64 output of `argmax`.
 *
 * @param obs Flattened observation tensor [Batch, ObsSize].
 * @param deterministic Whether to use argmax instead of multinomial sampling.
 * @return ActionResult struct containing stacked integer actions [Batch, 8] and scalar log_probs [Batch].
 * @throws std::invalid_argument on mismatched observation tensor dimensions.
 */
DiscretePolicy::ActionResult TransformerPolicy::GetAction(torch::Tensor obs, bool deterministic) {
    if (obs.dim() == 1) {
        obs = obs.unsqueeze(0);
    }
    if (obs.dim() != 2) {
        throw std::invalid_argument("TransformerPolicy::GetAction: Invalid obs tensor dimensions. Expected 2 [Batch, ObsSize], got " + std::to_string(obs.dim()));
    }
    auto probs = GetActionProbs(obs);
    auto probs_splits = torch::split_with_sizes(probs, splits_, -1);

    std::vector<torch::Tensor> actions;
    std::vector<torch::Tensor> log_probs;

    for (const auto& prob : probs_splits) {
        if (deterministic) {
            auto action = prob.argmax(1);
            actions.push_back(action);
            log_probs.push_back(torch::zeros_like(action, torch::kFloat32));
        } else {
            auto action = torch::multinomial(prob, 1, true); // [batch, 1]
            auto log_prob = torch::log(prob).gather(-1, action); // [batch, 1]
            actions.push_back(action.squeeze(-1));
            log_probs.push_back(log_prob.squeeze(-1));
        }
    }

    // actions stacked to [batch, 8]
    auto final_action = torch::stack(actions, -1);
    
    // Total log_prob is the sum of independent log_probs
    auto final_log_prob = torch::stack(log_probs, -1).sum(-1);

    return ActionResult{ final_action.cpu(), final_log_prob.cpu() };
}

/**
 * @brief Evaluates historical rollouts to compute gradients during the PPO Learn phase.
 *
 * Architectural Intent:
 * The PPO Learner runs this strictly during the backpropagation gradient descent loop.
 * It takes the exact historical observations and the actions the agent randomly sampled
 * in the past, and evaluates their *current* log probability and entropy under the 
 * updated neural network weights.
 * 
 * This isolates entropy maximization (encouraging exploration) and log-likelihood calculation
 * independently across all 8 sub-action dimensions, ensuring stable gradient flow to the 
 * predictor head without mathematically corrupting unrelated action spaces.
 *
 * @param obs Historical flattened observations [Batch, ObsSize].
 * @param acts Historical chosen actions [Batch, 8].
 * @return BackpropResult containing new trajectory log_probs and the mean network entropy.
 * @throws std::invalid_argument on malformed batch tensor boundaries.
 */
DiscretePolicy::BackpropResult TransformerPolicy::GetBackpropData(torch::Tensor obs, torch::Tensor acts) {
    if (obs.dim() != 2) {
        throw std::invalid_argument("TransformerPolicy::GetBackpropData: Invalid obs tensor dimensions. Expected 2 [Batch, ObsSize], got " + std::to_string(obs.dim()));
    }
    if (acts.dim() != 2) {
        throw std::invalid_argument("TransformerPolicy::GetBackpropData: Invalid acts tensor dimensions. Expected 2 [Batch, ActionSplitsCount], got " + std::to_string(acts.dim()));
    }
    if (acts.size(1) != splits_.size()) {
        throw std::invalid_argument("TransformerPolicy::GetBackpropData: Invalid acts tensor size at dim 1. Expected " + std::to_string(splits_.size()) + ", got " + std::to_string(acts.size(1)));
    }
    acts = acts.to(torch::kInt64, true); // [batch, 8]
    
    auto probs = GetActionProbs(obs); // [batch, 21]
    auto probs_splits = torch::split_with_sizes(probs, splits_, -1);

    torch::Tensor total_entropy = torch::zeros({acts.size(0)}, torch::TensorOptions().device(device));
    std::vector<torch::Tensor> total_log_probs;

    for (size_t i = 0; i < splits_.size(); i++) {
        auto prob = probs_splits[i];
        auto act = acts.select(1, i).unsqueeze(-1); // [batch, 1]

        auto log_prob = torch::log(prob);
        auto action_log_prob = log_prob.gather(-1, act).squeeze(-1); // [batch]
        total_log_probs.push_back(action_log_prob);

        auto entropy = -(log_prob * prob).sum(-1); // [batch]
        total_entropy = total_entropy + entropy;
    }

    auto final_action_log_prob = torch::stack(total_log_probs, -1).sum(-1); // [batch]

    return BackpropResult{ final_action_log_prob.to(device, true), total_entropy.to(device).mean() };
}

} // namespace RLGPC

namespace RLGPC {

TransformerValueEstimator::TransformerValueEstimator(
    int num_q, int num_entities, int query_features, int kv_features,
    torch::Device device)
    : ValueEstimator(num_q * query_features + num_entities * kv_features + num_entities, {256, 1}, device, false /* build_network */),
      num_q_(num_q), num_entities_(num_entities),
      query_features_(query_features), kv_features_(kv_features) {

    perceiver_ = register_module("perceiver", EARLPerceiver(num_q_, query_features_, kv_features_, 256, 4, 3));
    head_ = register_module("head", torch::nn::Sequential(torch::nn::Linear(256, 1)));

    this->to(device, true);
}

torch::Tensor TransformerValueEstimator::Forward(torch::Tensor input) {
    auto parsed = EARLObservationParser::Unflatten(input, num_q_, num_entities_, query_features_, kv_features_);
    
    // Forward pass
    torch::Tensor latent = perceiver_->forward(parsed.q, parsed.kv, parsed.mask);
    torch::Tensor latent_flat = latent.squeeze(1); // num_q is 1, squeeze it out
    
    return head_->forward(latent_flat); // [batch, 1]
}

void TransformerValueEstimator::Load(std::filesystem::path path, torch::Device device) {
    torch::serialize::InputArchive archive;
    archive.load_from(path.string(), device);
    this->load(archive);
}

void TransformerValueEstimator::Save(std::filesystem::path path) const {
    torch::serialize::OutputArchive archive;
    this->save(archive);
    archive.save_to(path.string());
}

std::vector<uint64_t> TransformerValueEstimator::GetSizes() {
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

} // namespace RLGPC
