#include "EARLPerceiver.h"
#include <ATen/autocast_mode.h>

namespace RLGPC {

EARLPerceiverBlockImpl::EARLPerceiverBlockImpl(int n_dims, int n_heads, int dim_feedforward, bool concatenate, bool use_norm) 
    : concatenate_(concatenate), use_norm_(use_norm) {
    
    if (dim_feedforward == 0) {
        dim_feedforward = 2 * n_dims;
    }

    attention_ = register_module("attention", torch::nn::MultiheadAttention(n_dims, n_heads));
    linear1_ = register_module("linear1", torch::nn::Linear((1 + concatenate_) * n_dims, dim_feedforward));
    linear2_ = register_module("linear2", torch::nn::Linear(dim_feedforward, n_dims));

    if (use_norm_) {
        norm1_ = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({n_dims})));
        norm2_ = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({n_dims})));
        norm3_ = register_module("norm3", torch::nn::LayerNorm(torch::nn::LayerNormOptions({n_dims})));
    }
}

/**
 * @brief Forward pass executing LayerNorm, Multi-Head Attention, and MLP Feedforward.
 *
 * LibTorch Integration:
 * Unlike Python PyTorch where `batch_first=True` is easily accessible, the C++ LibTorch 
 * `MultiheadAttention` module strictly requires `[Seq, Batch, Features]` tensors for performance
 * reasons. Therefore, we explicitly transpose the input queries and keys/values prior to attention
 * and transpose the output back to `[Batch, Seq, Features]`.
 * 
 * @param q Query tensor (Agent self-state features) of shape [Batch, Seq, Features].
 * @param kv Key/Value tensor (Other entities features) of shape [Batch, Seq, Features].
 * @param mask Boolean mask tensor of shape [Batch, Seq]. 'True' values indicate padded entities to be ignored.
 * @return Computed latent tensor of shape [Batch, Seq, Features].
 * @throws std::invalid_argument on mismatched shape dimensions or mismatched device allocations.
 */
torch::Tensor EARLPerceiverBlockImpl::forward(torch::Tensor q, torch::Tensor kv, const torch::Tensor& mask) {
    if (q.dim() != 3 || kv.dim() != 3) {
        throw std::invalid_argument("EARLPerceiverBlockImpl::forward: Invalid tensor dim. Expected 3 [Batch, Seq, Features]. Actual q: " + std::to_string(q.dim()) + ", kv: " + std::to_string(kv.dim()));
    }
    if (mask.dim() != 2) {
        throw std::invalid_argument("EARLPerceiverBlockImpl::forward: Invalid mask dim. Expected 2 [Batch, Seq]. Actual mask: " + std::to_string(mask.dim()));
    }
    if (q.size(0) != kv.size(0) || q.size(0) != mask.size(0)) {
        throw std::invalid_argument("EARLPerceiverBlockImpl::forward: Batch size mismatch. q: " + std::to_string(q.size(0)) + " kv: " + std::to_string(kv.size(0)) + " mask: " + std::to_string(mask.size(0)));
    }
    if (kv.size(1) != mask.size(1)) {
        throw std::invalid_argument("EARLPerceiverBlockImpl::forward: Seq len mismatch between kv and mask. kv: " + std::to_string(kv.size(1)) + " mask: " + std::to_string(mask.size(1)));
    }
    if (q.device() != kv.device() || q.device() != mask.device()) {
        throw std::invalid_argument("EARLPerceiverBlockImpl::forward: Device mismatch among inputs.");
    }

    torch::Tensor q_norm = use_norm_ ? norm1_->forward(q) : q;
    torch::Tensor kv_norm = use_norm_ ? norm2_->forward(kv) : kv;

    // MultiheadAttention expects [seq, batch, features] by default in C++
    torch::Tensor q_t = q_norm.transpose(0, 1);
    torch::Tensor kv_t = kv_norm.transpose(0, 1);

    // By forcing `false`, we allow PyTorch to use optimized SDPA/FlashAttention kernels,
    // which significantly speeds up the forward and backward pass of the attention layer.
    // Striding crashes and BFloat16 missing kernels have been resolved by disabling AMP.
    bool autocast_state = at::autocast::is_autocast_enabled(at::kCUDA);
    at::autocast::set_autocast_enabled(at::kCUDA, false);
    
    // Explicitly cast to FP32 to ensure the pre-compiled masked_fill dispatch doesn't crash on Half
    auto attn_res = attention_->forward(q_t.to(torch::kFloat32), kv_t.to(torch::kFloat32), kv_t.to(torch::kFloat32), mask, false);
    
    at::autocast::set_autocast_enabled(at::kCUDA, autocast_state);
    
    // Transpose output back to [batch, seq, features]
    torch::Tensor attn_out = std::get<0>(attn_res).transpose(0, 1);

    torch::Tensor x;
    if (concatenate_) {
        x = torch::cat({q, attn_out}, /*dim=*/-1);
    } else {
        x = q + attn_out;
    }

    torch::Tensor x_norm = use_norm_ ? norm3_->forward(x) : x;
    torch::Tensor fwd = linear2_->forward(torch::relu(linear1_->forward(x_norm)));

    return x + fwd;
}

EARLPerceiverImpl::EARLPerceiverImpl(int num_q, int query_features, int kv_features, int n_dims, int n_heads, int n_blocks) 
    : n_dims_(n_dims), n_entities_(num_q) {
    
    query_mlp_ = register_module("query_mlp", torch::nn::Sequential(
        torch::nn::Linear(query_features, n_dims_),
        torch::nn::ReLU(),
        torch::nn::Linear(n_dims_, n_dims_),
        torch::nn::ReLU()
    ));

    kv_mlp_ = register_module("kv_mlp", torch::nn::Sequential(
        torch::nn::Linear(kv_features, n_dims_),
        torch::nn::ReLU(),
        torch::nn::Linear(n_dims_, n_dims_),
        torch::nn::ReLU()
    ));

    blocks_ = register_module("blocks", torch::nn::ModuleList());
    for (int i = 0; i < n_blocks; i++) {
        blocks_->push_back(EARLPerceiverBlock(n_dims_, n_heads));
    }
}

/**
 * @brief Forward pass computing the aggregated latent feature representation of the environment.
 * 
 * Memory Management:
 * The MLP embeddings for queries and key/values are executed independently to project raw
 * spatial features into the high-dimensional latent space. The results are then sequentially
 * passed through `blocks_`. We leverage RAII auto-releasing variables to handle intermediate 
 * tensor allocations automatically throughout this deep network.
 *
 * @param q Query tensor (Agent state) of shape [Batch, NumQ, QueryFeatures].
 * @param kv Key/Value tensor (World state) of shape [Batch, NumEntities, KVFeatures].
 * @param mask Padding mask of shape [Batch, NumEntities].
 * @return Aggregated latent features of shape [Batch, NumQ, NDims].
 * @throws std::invalid_argument on invalid tensor shapes.
 */
torch::Tensor EARLPerceiverImpl::forward(torch::Tensor q, torch::Tensor kv, const torch::Tensor& mask) {
    if (q.dim() != 3 || kv.dim() != 3) {
        throw std::invalid_argument("EARLPerceiverImpl::forward: Invalid tensor dim. Expected 3 [Batch, Seq, Features]. Actual q: " + std::to_string(q.dim()) + ", kv: " + std::to_string(kv.dim()));
    }
    if (mask.dim() != 2) {
        throw std::invalid_argument("EARLPerceiverImpl::forward: Invalid mask dim. Expected 2 [Batch, Seq]. Actual mask: " + std::to_string(mask.dim()));
    }
    if (q.size(1) != n_entities_) {
        throw std::invalid_argument("EARLPerceiverImpl::forward: Invalid num_q in input. Expected " + std::to_string(n_entities_) + ", got " + std::to_string(q.size(1)));
    }
    // q: [batch, num_q, query_features]
    // kv: [batch, num_entities, kv_features]
    // mask: [batch, num_entities] boolean mask (True for padding/ignored)

    torch::Tensor q_emb = query_mlp_->forward(q);
    torch::Tensor kv_emb = kv_mlp_->forward(kv);

    for (const auto& block : *blocks_) {
        q_emb = block->as<EARLPerceiverBlockImpl>()->forward(q_emb, kv_emb, mask);
    }
    
    return q_emb; // shape: [batch, num_q, n_dims]
}

} // namespace RLGPC
