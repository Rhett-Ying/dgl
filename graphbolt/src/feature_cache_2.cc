#include "./feature_cache_2.h"

namespace graphbolt {
namespace storage {

FeatureCache2::FeatureCache2(
    const std::vector<int64_t>& shape, torch::ScalarType dtype)
    : key_cache_(shape[0]) {
  tensor_ = torch::empty(shape, torch::TensorOptions().dtype(dtype));
  index_map_.reserve(shape[0]);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> FeatureCache2::Query(
    torch::Tensor indices) {
  // data, found_keys, missing_keys.
  auto positions = key_cache_.Query(indices);
  if (positions.size() == 0) {
    return std::make_tuple(torch::empty({0}), torch::empty({0}), indices);
  }
  return std::make_tuple(
      torch::empty({0}), torch::empty({0}), torch::empty({0}));
}

torch::Tensor FeatureCache2::Replace(
    torch::Tensor indices, torch::Tensor values) {
  // Update the tensor with the new values. just iterate over the indices.
  // randomly pick a palce from index_map_ and update the tensor_ with the
  // values.
  auto indices_ptr = indices.data_ptr<int64_t>();
  auto values_ptr = values.data_ptr<float>();
  for (int i = 0; i < indices.size(0); i++) {
    int64_t position = key_cache_.Query(indices_ptr[i]);
    if (position == -1) {
      // already exists in the cache.
      continue;
    }
    // copy value to the tensor.
    tensor_[position] = values_ptr[i];
  }
  return torch::empty({0});
}
}  // namespace storage
}  // namespace graphbolt