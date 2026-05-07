#pragma once

#include <torch/torch.h>

#include <RLGymPPO_CPP/Framework.h>

namespace RLGPC {

/**
 * @brief A single Cross-Attention Transformer block for the EARLPerceiver.
 *
 * This block applies Multi-Head Attention where the queries (agent self-state) attend 
 * to keys/values (all other entities/pads in the match). It utilizes `torch::nn::MultiheadAttention`
 * internally and manages the memory overhead of transposing sequences between Batch-First 
 * and Sequence-First formats required by LibTorch.
 */
struct RG_IMEXPORT EARLPerceiverBlockImpl : torch::nn::Module {
    bool concatenate_;
    bool use_norm_;
    torch::nn::MultiheadAttention attention_{nullptr};
    torch::nn::Linear linear1_{nullptr};
    torch::nn::Linear linear2_{nullptr};
    torch::nn::LayerNorm norm1_{nullptr};
    torch::nn::LayerNorm norm2_{nullptr};
    torch::nn::LayerNorm norm3_{nullptr};

    EARLPerceiverBlockImpl(int n_dims, int n_heads, int dim_feedforward = 0, bool concatenate = false, bool use_norm = true);

    torch::Tensor forward(torch::Tensor q, torch::Tensor kv, const torch::Tensor& mask = {});
};
TORCH_MODULE(EARLPerceiverBlock);

/**
 * @brief The full EARLPerceiver architecture replacing the legacy MLP feature extractor.
 *
 * Architectural Intent:
 * The Rocket League environment is highly dynamic, containing 1 to N cars, ball, and boost 
 * pads. A standard MLP requires zero-padding missing entities, causing mathematically unsound
 * behavior as the network tries to map 0s. The Perceiver utilizes Cross-Attention to dynamically
 * aggregate features from an arbitrary number of entities, using a Boolean mask to completely 
 * ignore non-existent or out-of-bounds entities, ensuring perfectly scale-invariant feature extraction.
 */
struct RG_IMEXPORT EARLPerceiverImpl : torch::nn::Module {
    int n_dims_;
    int n_entities_;
    torch::nn::Sequential query_mlp_{nullptr};
    torch::nn::Sequential kv_mlp_{nullptr};
    torch::nn::ModuleList blocks_{nullptr};

    EARLPerceiverImpl(int num_q, int query_features, int kv_features, int n_dims = 256, int n_heads = 4, int n_blocks = 3);

    torch::Tensor forward(torch::Tensor q, torch::Tensor kv, const torch::Tensor& mask = {});
};
TORCH_MODULE(EARLPerceiver);

} // namespace RLGPC
