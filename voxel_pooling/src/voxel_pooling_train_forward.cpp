#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <torch/extension.h>
#include <torch/serialize/tensor.h>

#include <vector>
#define CHECK_CUDA(x) \
  TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) \
  TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) \
  CHECK_CUDA(x);       \
  CHECK_CONTIGUOUS(x)

int voxel_pooling_train_forward_wrapper(int batch_size, int num_points, int num_features, int num_channels, 
                                        int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth,
                                        at::Tensor geom_xyz_tensor,
                                        at::Tensor input_features_tensor,
                                        at::Tensor depth_tensor,
                                        at::Tensor output_features_tensor,
                                        at::Tensor pos_memo_tensor);

int voxel_pooling_train_backward_wrapper(int batch_size, int num_points, int num_features, int num_channels, 
                                        int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth,
                                        at::Tensor pos_memo_tensor,
                                        at::Tensor input_features_tensor,
                                        at::Tensor depth_tensor,
                                        at::Tensor grad_input_features_tensor,
                                        at::Tensor grad_depth_tensor,
                                        at::Tensor grad_output_features_tensor);

void voxel_pooling_train_forward_kernel_launcher(
    int batch_size, int num_points, int num_features, int num_channels, 
    int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth,
    const int *geom_xyz, const float *input_features, const float *depth, float *output_features, int *pos_memo,
    cudaStream_t stream);

void voxel_pooling_train_backward_kernel_launcher(
    int batch_size, int num_points, int num_features, int num_channels, 
    int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth,
    const int *pos_memo, const float *input_features, const float *depth, float *grad_input_features, float *grad_depth, const float *grad_output_features,
    cudaStream_t stream);

int voxel_pooling_train_forward_wrapper(int batch_size, int num_points, int num_features, int num_channels, 
                                        int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth,
                                        at::Tensor geom_xyz_tensor,
                                        at::Tensor input_features_tensor,
                                        at::Tensor depth_tensor,
                                        at::Tensor output_features_tensor,
                                        at::Tensor pos_memo_tensor) {
  CHECK_INPUT(geom_xyz_tensor);
  CHECK_INPUT(input_features_tensor);
  CHECK_INPUT(depth_tensor);
  const int *geom_xyz = geom_xyz_tensor.data_ptr<int>();
  int *pos_memo = pos_memo_tensor.data_ptr<int>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

  const float *input_features = input_features_tensor.data_ptr<float>();
  const float *depth = depth_tensor.data_ptr<float>();
  float *output_features = output_features_tensor.data_ptr<float>();
  voxel_pooling_train_forward_kernel_launcher(
      batch_size, num_points, num_features, num_channels, num_voxel_x, num_voxel_y,
      num_voxel_z, num_depth, geom_xyz, input_features, depth, output_features, pos_memo,
      stream);

  return 1;
}

int voxel_pooling_train_backward_wrapper(int batch_size, int num_points, int num_features, int num_channels, 
                                        int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth,
                                        at::Tensor pos_memo_tensor,
                                        at::Tensor input_features_tensor,
                                        at::Tensor depth_tensor,
                                        at::Tensor grad_input_features_tensor,
                                        at::Tensor grad_depth_tensor,
                                        at::Tensor grad_output_features_tensor)
{
  CHECK_INPUT(pos_memo_tensor);
  CHECK_INPUT(input_features_tensor);
  CHECK_INPUT(depth_tensor);
  CHECK_INPUT(grad_input_features_tensor);
  CHECK_INPUT(grad_depth_tensor);
  CHECK_INPUT(grad_output_features_tensor);
  const int *pos_memo = pos_memo_tensor.data_ptr<int>();
  const float *input_features = input_features_tensor.data_ptr<float>();
  const float *depth = depth_tensor.data_ptr<float>();
  float *grad_input_features = grad_input_features_tensor.data_ptr<float>();
  float *grad_depth = grad_depth_tensor.data_ptr<float>();
  const float *grad_output_features = grad_output_features_tensor.data_ptr<float>();

  cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
  voxel_pooling_train_backward_kernel_launcher(batch_size, num_points, num_features, num_channels, num_voxel_x, num_voxel_y, num_voxel_z, num_depth, 
        pos_memo, input_features, depth, grad_input_features, grad_depth, grad_output_features, stream);
  return 1;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("voxel_pooling_train_forward_wrapper",
        &voxel_pooling_train_forward_wrapper,
        "voxel_pooling_train_forward_wrapper");
  m.def("voxel_pooling_train_backward_wrapper",
        &voxel_pooling_train_backward_wrapper,
        "voxel_pooling_train_backward_wrapper");
}
