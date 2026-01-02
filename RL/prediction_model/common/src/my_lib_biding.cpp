#include <pybind11/pybind11.h>
#include <ATen/ATen.h>

// Function declarations
int jagged_argmax_forward(const at::Tensor& values, const at::Tensor& prefix_sum, at::Tensor& output);
int jagged_max_forward(const at::Tensor& values, const at::Tensor& prefix_sum, at::Tensor& vmax, at::Tensor& idxes);
int jagged_log_softmax_forward(const at::Tensor& logits, const at::Tensor& prefix_sum, at::Tensor& output);
int jagged_log_softmax_backward(const at::Tensor& output, const at::Tensor& grad_output, const at::Tensor& prefix_sum, at::Tensor& grad_input);
int graph_laplacian_norm(const at::Tensor& indices, const at::Tensor& values, at::Tensor& norm);
int graph_degree_norm(const at::Tensor& indices, const at::Tensor& values, at::Tensor& norm);

int jagged_log_softmax_forward_cuda(at::Tensor logits, at::Tensor prefix_sum, at::Tensor output);
int jagged_log_softmax_backward_cuda(at::Tensor output, at::Tensor grad_output, at::Tensor prefix_sum, at::Tensor grad_input);
int jagged_argmax_forward_cuda(at::Tensor values, at::Tensor prefix_sum, at::Tensor output);
int jagged_max_forward_cuda(at::Tensor values, at::Tensor prefix_sum, at::Tensor vmax, at::Tensor idxes);
int graph_laplacian_norm_cuda(at::Tensor indices, at::Tensor values, at::Tensor norm);
int graph_degree_norm_cuda(at::Tensor indices, at::Tensor values, at::Tensor norm);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // Bindings for CPU functions
    m.def("jagged_log_softmax_forward", &jagged_log_softmax_forward, "Jagged Log Softmax Forward (CPU)");
    m.def("jagged_log_softmax_backward", &jagged_log_softmax_backward, "Jagged Log Softmax Backward (CPU)");
    m.def("jagged_argmax_forward", &jagged_argmax_forward, "Jagged Argmax Forward (CPU)");
    m.def("jagged_max_forward", &jagged_max_forward, "Jagged Max Forward (CPU)");
    m.def("graph_laplacian_norm", &graph_laplacian_norm, "Graph Laplacian Norm (CPU)");
    m.def("graph_degree_norm", &graph_degree_norm, "Graph Degree Norm (CPU)");

    // Bindings for CUDA functions
    m.def("jagged_log_softmax_forward_cuda", &jagged_log_softmax_forward_cuda, "Jagged Log Softmax Forward (CUDA)");
    m.def("jagged_log_softmax_backward_cuda", &jagged_log_softmax_backward_cuda, "Jagged Log Softmax Backward (CUDA)");
    m.def("jagged_argmax_forward_cuda", &jagged_argmax_forward_cuda, "Jagged Argmax Forward (CUDA)");
    m.def("jagged_max_forward_cuda", &jagged_max_forward_cuda, "Jagged Max Forward (CUDA)");
    m.def("graph_laplacian_norm_cuda", &graph_laplacian_norm_cuda, "Graph Laplacian Norm (CUDA)");
    m.def("graph_degree_norm_cuda", &graph_degree_norm_cuda, "Graph Degree Norm (CUDA)");
}
