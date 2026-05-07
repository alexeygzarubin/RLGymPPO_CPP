#include "ControlsPredictorDiscrete.h"

namespace RLGPC {

ControlsPredictorDiscreteImpl::ControlsPredictorDiscreteImpl(int n_dims, std::vector<int64_t> splits) 
    : splits_(std::move(splits)) {
    
    int64_t sum = 0;
    for (int64_t s : splits_) {
        sum += s;
    }
    
    linear_ = register_module("linear", torch::nn::Linear(n_dims, sum));
}

/**
 * @brief Forward pass resolving independent action bins as discrete tensors.
 *
 * @param emb The latent state embedding tensor of shape [Batch, Features].
 * @return A vector of tensors where each tensor represents unscaled logits for a distinct action axis.
 * @throws std::invalid_argument on mismatched shape dimensions.
 */
std::vector<torch::Tensor> ControlsPredictorDiscreteImpl::forward(torch::Tensor emb) {
    if (emb.dim() != 2) {
        throw std::invalid_argument("ControlsPredictorDiscreteImpl::forward: Invalid tensor dim. Expected 2 [Batch, Features], got " + std::to_string(emb.dim()));
    }
    torch::Tensor actions = linear_->forward(emb);
    // Split the final linear output into multiple action tensors based on the bins
    return torch::split_with_sizes(actions, splits_, /*dim=*/-1);
}

/**
 * @brief Accelerated forward pass maintaining a flattened output space.
 *
 * Performance considerations:
 * Re-allocating the vector arrays through `torch::split_with_sizes` is computationally 
 * expensive during large batched operations in the PPO learning loop. Returning the flat 
 * tensor directly allows upstream architectures (like TransformerPolicy) to optimize Softmax
 * chunking dynamically in bulk, bypassing unnecessary nested tensor structs.
 *
 * @param emb The latent state embedding tensor of shape [Batch, Features].
 * @return Flat contiguous tensor of unscaled logits, shape [Batch, TotalActionSize].
 * @throws std::invalid_argument on mismatched shape dimensions.
 */
torch::Tensor ControlsPredictorDiscreteImpl::forward_flat(torch::Tensor emb) {
    if (emb.dim() != 2) {
        throw std::invalid_argument("ControlsPredictorDiscreteImpl::forward_flat: Invalid tensor dim. Expected 2 [Batch, Features], got " + std::to_string(emb.dim()));
    }
    // Returns the unsplit concatenated logits for easier batched softmax
    return linear_->forward(emb);
}

} // namespace RLGPC
