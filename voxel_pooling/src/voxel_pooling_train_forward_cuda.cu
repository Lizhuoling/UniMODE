#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

template <typename T>
__global__ void voxel_pooling_train_forward_kernel(
    int num_proj, int num_features, int num_channels, int num_depth, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const T *input_features, const T *depth, T *output_features) 
{

  const int blk_idx = blockIdx.x;
  const int thd_idx = threadIdx.x;
  const int pt_idx = blk_idx * blockDim.x + thd_idx;
  if (pt_idx >= num_proj) 
  {
    return;
  } 
  else 
  {
    const int x = geom_xyz[pt_idx * 6];
    const int y = geom_xyz[pt_idx * 6 + 1];
    const int z = geom_xyz[pt_idx * 6 + 2];
    const int batch_idx = geom_xyz[pt_idx * 6 + 3];
    const int depth_idx = geom_xyz[pt_idx * 6 + 4];
    const int feature_idx = geom_xyz[pt_idx * 6 + 5];

    const int grid_id = batch_idx * num_voxel_z * num_voxel_y * num_voxel_x + z * num_voxel_y * num_voxel_x + y * num_voxel_x + x;

    for (int channel_idx = 0; channel_idx < num_channels; channel_idx++) 
    {
      atomicAdd(
          &output_features[grid_id * num_channels + channel_idx],
          input_features[(batch_idx * num_channels + channel_idx) * num_features + feature_idx] * depth[(batch_idx * num_depth + depth_idx) * num_features + feature_idx]
      );
    }
  }
}

__global__ void  voxel_pooling_train_backward_kernel(
    int num_proj, int num_features, int num_channels, int num_depth, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const float *input_features, const float *depth, float *grad_input_features, float *grad_depth, const float *grad_output_features)
{
  // Each thread process only one channel of one voxel.
  int blk_idx = blockIdx.x;
  int thd_idx = threadIdx.x;
  int pt_idx = blk_idx * blockDim.x + thd_idx;
  if (pt_idx >= num_proj) 
  {
    return;
  } 
  else 
  {
    const int x = geom_xyz[pt_idx * 6];
    const int y = geom_xyz[pt_idx * 6 + 1];
    const int z = geom_xyz[pt_idx * 6 + 2];
    const int batch_idx = geom_xyz[pt_idx * 6 + 3];
    const int depth_idx = geom_xyz[pt_idx * 6 + 4];
    const int feature_idx = geom_xyz[pt_idx * 6 + 5];

    const int grid_id = batch_idx * num_voxel_z * num_voxel_y * num_voxel_x + z * num_voxel_y * num_voxel_x + y * num_voxel_x + x;

    for (int channel_idx = 0; channel_idx < num_channels; channel_idx++) 
    {
      atomicAdd(
        &grad_input_features[(batch_idx * num_channels + channel_idx) * num_features + feature_idx],
          grad_output_features[grid_id * num_channels + channel_idx] * 
          depth[(batch_idx * num_depth + depth_idx) * num_features + feature_idx]);

      atomicAdd(
        &grad_depth[(batch_idx * num_depth + depth_idx) * num_features + feature_idx],
          grad_output_features[grid_id * num_channels + channel_idx] * 
          input_features[(batch_idx * num_channels + channel_idx) * num_features + feature_idx]);
    }
  }
}

void voxel_pooling_train_forward_kernel_launcher( 
    int num_proj, int num_features, int num_channels, int num_depth, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const float *input_features, const float *depth, float *output_features,
    cudaStream_t stream) {
  cudaError_t err;

  dim3 blocks(DIVUP(num_proj, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  voxel_pooling_train_forward_kernel<<<blocks, threads, 0, stream>>>(
      num_proj, num_features, num_channels, num_depth, num_voxel_x, num_voxel_y, num_voxel_z, 
      geom_xyz, input_features, depth, output_features);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void voxel_pooling_train_backward_kernel_launcher(
    int num_proj, int num_features, int num_channels, int num_depth, int num_voxel_x, int num_voxel_y, int num_voxel_z,
    const int *geom_xyz, const float *input_features, const float *depth, float *grad_input_features, float *grad_depth, const float *grad_output_features,
    cudaStream_t stream){
  cudaError_t err;

  dim3 blocks(DIVUP(num_proj, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  voxel_pooling_train_backward_kernel<<<blocks, threads, 0, stream>>>(
      num_proj, num_features, num_channels, num_depth, num_voxel_x, num_voxel_y, num_voxel_z, 
      geom_xyz, input_features, depth, grad_input_features, grad_depth, grad_output_features);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
