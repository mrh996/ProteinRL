#include <ATen/ATen.h>
#include <c10/cuda/CUDAStream.h>
#include "custom_kernel.h"
//#include <pybind11/pybind11.h>

//namespace py = pybind11;

int jagged_log_softmax_forward_cuda(at::Tensor logits, at::Tensor prefix_sum, at::Tensor output) {
    logits = logits.contiguous();
    output.resize_as_(logits);

    float* input_data_base = logits.data_ptr<float>();
    long* ps = prefix_sum.data_ptr<int64_t>();
    float* output_data_base = output.data_ptr<float>();

    int bsize = prefix_sum.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    HostSoftMaxForward(stream, input_data_base, output_data_base, ps, bsize);

    return 1;
}

int jagged_log_softmax_backward_cuda(at::Tensor output, at::Tensor grad_output, at::Tensor prefix_sum, at::Tensor grad_input) {
    output = output.contiguous();
    grad_output = grad_output.contiguous();

    grad_input.resize_as_(grad_output);
    float* output_data_base = output.data_ptr<float>();
    float* gradOutput_data_base = grad_output.data_ptr<float>();
    int64_t* ps = prefix_sum.data_ptr<int64_t>();
    float* gradInput_data_base = grad_input.data_ptr<float>();

    int bsize = prefix_sum.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    HostSoftMaxBackward(stream, gradOutput_data_base, gradInput_data_base, output_data_base, ps, bsize);

    return 1;
}

int jagged_argmax_forward_cuda(at::Tensor values, at::Tensor prefix_sum, at::Tensor output) {
    values = values.contiguous();
    output.resize_as_(prefix_sum);

    float* input_data_base = values.data_ptr<float>();
    int64_t* ps = prefix_sum.data_ptr<int64_t>();
    int64_t* output_data_base = output.data_ptr<int64_t>();

    int bsize = prefix_sum.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    HostArgmaxForward(stream, input_data_base, output_data_base, ps, bsize);

    return 1;
}

int jagged_max_forward_cuda(at::Tensor values, at::Tensor prefix_sum, at::Tensor vmax, at::Tensor idxes) {
    values = values.contiguous();
    idxes.resize_(prefix_sum.sizes());
    vmax.resize_(prefix_sum.sizes());

    float* input_data_base = values.data_ptr<float>();
    int64_t* ps = prefix_sum.data_ptr<int64_t>();
    int64_t* p_i = idxes.data_ptr<int64_t>();
    float* p_maxv = vmax.data_ptr<float>();

    int bsize = prefix_sum.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    HostMaxForward(stream, input_data_base, p_maxv, p_i, ps, bsize);

    return 1;
}

int graph_laplacian_norm_cuda(at::Tensor indices, at::Tensor values, at::Tensor norm) {
    uint64_t nnz = values.size(0);
    int64_t* row_indices = indices.data_ptr<int64_t>();
    int64_t* col_indices = row_indices + indices.stride(0);
    float* p_v = values.data_ptr<float>();
    float* p_norm = norm.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream();

    HostGLapNorm(stream, row_indices, col_indices, p_v, p_norm, nnz);
    return 1;
}

int graph_degree_norm_cuda(at::Tensor indices, at::Tensor values, at::Tensor norm) {
    uint64_t nnz = values.size(0);
    int64_t* row_indices = indices.data_ptr<int64_t>();
    float* p_v = values.data_ptr<float>();
    float* p_norm = norm.data_ptr<float>();

    auto stream = at::cuda::getCurrentCUDAStream();

    HostGDegreeNorm(stream, row_indices, p_v, p_norm, nnz);
    return 1;
}

//PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
//    m.def("jagged_log_softmax_forward_cuda", &jagged_log_softmax_forward_cuda, "Jagged Log Softmax Forward (CUDA)");
//    m.def("jagged_log_softmax_backward_cuda", &jagged_log_softmax_backward_cuda, "Jagged Log Softmax Backward (CUDA)");
 //   m.def("jagged_argmax_forward_cuda", &jagged_argmax_forward_cuda, "Jagged Argmax Forward (CUDA)");
 //   m.def("jagged_max_forward_cuda", &jagged_max_forward_cuda, "Jagged Max Forward (CUDA)");
 //   m.def("graph_laplacian_norm_cuda", &graph_laplacian_norm_cuda, "Graph Laplacian Norm (CUDA)");
 //   m.def("graph_degree_norm_cuda", &graph_degree_norm_cuda, "Graph Degree Norm (CUDA)");
//}
