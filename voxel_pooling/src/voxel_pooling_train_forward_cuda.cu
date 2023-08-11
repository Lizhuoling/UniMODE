#include <cuda_fp16.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define THREADS_PER_BLOCK 1024
#define DIVUP(m, n) ((m) / (n) + ((m) % (n) > 0))

template <typename T>
__global__ void voxel_pooling_train_forward_kernel(
    int batch_size, int num_points, int num_features, int num_channels, 
    int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth, 
    const int *geom_xyz, const T *input_features, const T *depth, T *output_features) 
{

  const int blk_idx = blockIdx.x;
  const int thd_idx = threadIdx.x;
  const int pt_idx = blk_idx * blockDim.x + thd_idx;
  if (pt_idx >= batch_size * num_points) 
  {
    return;
  } 
  else 
  {
    const int batch_idx = pt_idx / num_points;  // Batch idx 
    const int tmp_idx = pt_idx % num_points;  // The idx in DHW dims
    const int depth_idx = tmp_idx / num_features; // The D idx
    const int feature_idx = tmp_idx % num_features; // The HW idx

    const int x = geom_xyz[pt_idx * 3];
    const int y = geom_xyz[pt_idx * 3 + 1];
    const int z = geom_xyz[pt_idx * 3 + 2];
    // if coord of current voxel is out of boundary, return.

    if (x < 0 || x >= num_voxel_x || y < 0 || y >= num_voxel_y || z < 0 || z >= num_voxel_z) {
      return;
    }

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
    int batch_size, int num_points, int num_features, int num_channels, 
    int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth,
    const int *geom_xyz, const float *input_features, const float *depth, float *grad_input_features, float *grad_depth, const float *grad_output_features)
{
  // Each thread process only one channel of one voxel.
  int blk_idx = blockIdx.x;
  int thd_idx = threadIdx.x;
  int pt_idx = blk_idx * blockDim.x + thd_idx;
  if (pt_idx >= batch_size * num_points) 
  {
    return;
  } 
  else 
  {
    const int batch_idx = pt_idx / num_points;  // Batch idx 
    const int tmp_idx = pt_idx % num_points;  // The idx in DHW dims
    const int depth_idx = tmp_idx / num_features; // The D idx
    const int feature_idx = tmp_idx % num_features; // The HW idx

    const int x = geom_xyz[pt_idx * 3];
    const int y = geom_xyz[pt_idx * 3 + 1];
    const int z = geom_xyz[pt_idx * 3 + 2];
    // if point is not used, return.
    if (x < 0 || x >= num_voxel_x || y < 0 || y >= num_voxel_y || z < 0 || z >= num_voxel_z) {
      return;
    }

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
    int batch_size, int num_points, int num_features, int num_channels, 
    int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth,
    const int *geom_xyz, const float *input_features, const float *depth, float *output_features,
    cudaStream_t stream) {
  cudaError_t err;

  dim3 blocks(DIVUP(batch_size * num_points, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  voxel_pooling_train_forward_kernel<<<blocks, threads, 0, stream>>>(
      batch_size, num_points, num_features, num_channels, num_voxel_x, num_voxel_y,
      num_voxel_z, num_depth, geom_xyz, input_features, depth, output_features);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}

void voxel_pooling_train_backward_kernel_launcher(
    int batch_size, int num_points, int num_features, int num_channels, 
    int num_voxel_x, int num_voxel_y, int num_voxel_z, int num_depth,
    const int *geom_xyz, const float *input_features, const float *depth, float *grad_input_features, float *grad_depth, const float *grad_output_features,
    cudaStream_t stream){
  cudaError_t err;

  dim3 blocks(DIVUP(batch_size * num_points, THREADS_PER_BLOCK)); // blockIdx.x(col), blockIdx.y(row)
  dim3 threads(THREADS_PER_BLOCK);

  voxel_pooling_train_backward_kernel<<<blocks, threads, 0, stream>>>(
      batch_size, num_points, num_features, num_channels, num_voxel_x, num_voxel_y,
      num_voxel_z, num_depth, geom_xyz, input_features, depth, grad_input_features, grad_depth, grad_output_features);
  err = cudaGetLastError();
  if (cudaSuccess != err) {
    fprintf(stderr, "CUDA kernel failed : %s\n", cudaGetErrorString(err));
    exit(-1);
  }
}
