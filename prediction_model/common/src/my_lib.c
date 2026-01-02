#include <ATen/ATen.h>
#include <vector>
#include <cassert>
#include <cmath>
#include <assert.h>
#include <cfloat>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

int jagged_argmax_forward(const at::Tensor& values, const at::Tensor& prefix_sum, at::Tensor& output) {
    auto values_cont = values.contiguous();
    auto prefix_sum_cont = prefix_sum.contiguous();

    const float *input_data_base = values_cont.data_ptr<float>();
    const long *ps = prefix_sum_cont.data_ptr<long>();
    long *p_out = output.data_ptr<long>();
    long bsize = prefix_sum.size(0);
    long i, d;

    for (i = 0; i < bsize; i++) {
        long offset = (i == 0) ? 0 : ps[i - 1];
        long n_ele = ps[i] - offset;
        float max_input = -FLT_MAX;
        long max_id = -1;
        for (d = 0; d < n_ele; d++) {
            float value = input_data_base[offset + d];
            if (value > max_input) {
                max_input = value;
                max_id = d;
            }
        }
        assert(max_id >= 0);
        p_out[i] = max_id;
    }
    return 1;
}


int jagged_max_forward(std::vector<float>& values, std::vector<long>& prefix_sum, std::vector<float>& vmax, std::vector<long>& idxes)
{
    int64_t inputsize = prefix_sum.size();
    idxes.resize(inputsize);
    vmax.resize(inputsize);

    float* input_data_base = values.data();
    long* ps = prefix_sum.data();
    float* p_maxv = vmax.data();
    long* p_i = idxes.data();

    long bsize = static_cast<long>(prefix_sum.size());
    long i, d;

    #pragma omp parallel for private(i, d)
    for (i = 0; i < bsize; i++)
    {
        long offset = (i == 0) ? 0 : ps[i - 1];
        long n_ele = ps[i] - offset;

        float* input_data = input_data_base + offset;

        float max_input = -FLT_MAX;
        long max_id = -1;
        for (d = 0; d < n_ele; d++)
        {
            if (input_data[d] > max_input)
            {
                max_input = input_data[d];
                max_id = d;
            }
        }
        assert(max_id >= 0);
        p_i[i] = max_id;
        p_maxv[i] = max_input;
    }

    return 1;
}


int jagged_log_softmax_forward(const at::Tensor& logits, const at::Tensor& prefix_sum, at::Tensor& output) {
    auto logits_cont = logits.contiguous();
    output.resize_as_(logits_cont);

    const float *logits_data = logits_cont.data_ptr<float>();
    const long *ps = prefix_sum.data_ptr<long>();
    float *output_data = output.data_ptr<float>();

    long bsize = prefix_sum.size(0);
    for (long i = 0; i < bsize; i++) {
        long offset = (i == 0) ? 0 : ps[i - 1];
        long n_ele = ps[i] - offset;
        float max_logit = *std::max_element(logits_data + offset, logits_data + offset + n_ele);
        float sum_exp = 0.0;
        for (long j = 0; j < n_ele; j++) {
            sum_exp += std::exp(logits_data[offset + j] - max_logit);
        }
        float log_sum_exp = std::log(sum_exp) + max_logit;
        for (long j = 0; j < n_ele; j++) {
            output_data[offset + j] = logits_data[offset + j] - log_sum_exp;
        }
    }
    return 1;
}

int jagged_log_softmax_backward(const at::Tensor& output, const at::Tensor& grad_output, const at::Tensor& prefix_sum, at::Tensor& grad_input) {
    auto output_cont = output.contiguous();
    auto grad_output_cont = grad_output.contiguous();
    grad_input.resize_as_(output_cont);

    const float *output_data = output_cont.data_ptr<float>();
    const float *grad_output_data = grad_output_cont.data_ptr<float>();
    float *grad_input_data = grad_input.data_ptr<float>();
    const long *ps = prefix_sum.data_ptr<long>();

    long bsize = prefix_sum.size(0);
    for (long i = 0; i < bsize; i++) {
        long offset = (i == 0) ? 0 : ps[i - 1];
        long n_ele = ps[i] - offset;
        float sum = 0.0;
        for (long j = 0; j < n_ele; j++) {
            sum += grad_output_data[offset + j];
        }
        for (long j = 0; j < n_ele; j++) {
            grad_input_data[offset + j] = grad_output_data[offset + j] - std::exp(output_data[offset + j]) * sum;
        }
    }
    return 1;
}

int graph_laplacian_norm(const at::Tensor& indices, const at::Tensor& values, at::Tensor& norm) {
    assert(indices.is_contiguous());
    assert(values.is_contiguous());
    assert(norm.is_contiguous());

    int64_t nnz = values.size(0);
    const long* row_indices = indices.data_ptr<long>();
    const long* col_indices = row_indices + indices.stride(0);
    float* p_v = values.data_ptr<float>();
    const float* p_norm = norm.data_ptr<float>();

    #pragma omp parallel for
    for (int64_t i = 0; i < nnz; i++) {
        float norm_value = p_norm[row_indices[i]] * p_norm[col_indices[i]];
        p_v[i] /= norm_value;
    }

    return 1;
}

int graph_degree_norm(const at::Tensor& indices, const at::Tensor& values, at::Tensor& norm) {
    assert(indices.is_contiguous());
    assert(values.is_contiguous());
    assert(norm.is_contiguous());

    int64_t nnz = values.size(0);
    const long* row_indices = indices.data_ptr<long>();
    float* p_v = values.data_ptr<float>();
    const float* p_norm = norm.data_ptr<float>();

    #pragma omp parallel for
    for (int64_t i = 0; i < nnz; i++) {
        float norm_value = p_norm[row_indices[i]];
        p_v[i] /= norm_value;
    }

    return 1;
}
PYBIND11_MODULE(my_lib, m) {
    m.def("jagged_log_softmax_forward", &jagged_log_softmax_forward, "Jagged Log Softmax Forward (CPU)");
    m.def("jagged_log_softmax_backward", &jagged_log_softmax_backward, "Jagged Log Softmax Backward (CPU)");
    m.def("jagged_argmax_forward", &jagged_argmax_forward, "Jagged Argmax Forward (CPU)");
    m.def("jagged_max_forward", &jagged_max_forward, "Jagged Max Forward (CPU)");
    m.def("graph_laplacian_norm", &graph_laplacian_norm, "Graph Laplacian Norm (CPU)");
    m.def("graph_degree_norm", &graph_degree_norm, "Graph Degree Norm (CPU)");
}
