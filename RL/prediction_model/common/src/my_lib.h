#pragma once
#include <ATen/ATen.h>
#include <vector>

int jagged_log_softmax_forward(const at::Tensor& logits, const at::Tensor& prefix_sum, at::Tensor& output);

int jagged_log_softmax_backward(const at::Tensor& output, const at::Tensor& grad_output, const at::Tensor& prefix_sum, at::Tensor& grad_input);

int jagged_argmax_forward(const at::Tensor& values, const at::Tensor& prefix_sum, at::Tensor& output);

int jagged_max_forward(const std::vector<float>& values, const std::vector<long>& prefix_sum, std::vector<float>& vmax, std::vector<long>& idxes);

int graph_laplacian_norm(const at::Tensor& indices, const at::Tensor& values, at::Tensor& norm);

int graph_degree_norm(const at::Tensor& indices, const at::Tensor& values, at::Tensor& norm);