#pragma once

#include <torch/torch.h>
#include <vector>

#include <RLGymPPO_CPP/Framework.h>

namespace RLGPC {

/**
 * @brief Predicts Rocket League controls across multiple independent discrete action bins.
 *
 * This module acts as the final decision head for the policy network. It takes the highly
 * compressed latent representation from the Perceiver and projects it into a flattened
 * linear space before selectively splitting it into the configured independent discrete
 * probability bins (e.g., Throttle, Steer, Jump, Boost, etc.).
 */
struct RG_IMEXPORT ControlsPredictorDiscreteImpl : torch::nn::Module {
    std::vector<int64_t> splits_;
    torch::nn::Linear linear_{nullptr};

    ControlsPredictorDiscreteImpl(int n_dims, std::vector<int64_t> splits = {3, 3, 3, 3, 3, 2, 2, 2});

    std::vector<torch::Tensor> forward(torch::Tensor emb);
    torch::Tensor forward_flat(torch::Tensor emb);
};
TORCH_MODULE(ControlsPredictorDiscrete);

} // namespace RLGPC
