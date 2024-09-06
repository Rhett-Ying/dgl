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
  // Retrieve the found keys and missing keys from key_cache_.
  auto [found_keys, missing_keys] = key_cache_.Query(indices);

  // read data from tensor_.
  auto data =
      torch::empty({found_keys.size(), tensor_.size(1)}, tensor_.options());
  for (int64_t i = 0; i < found_keys.size(); i++) {
    data[i] = tensor_[index_map_[found_keys[i]]];
  }

  // return found_keys, missing_keys as tensor.
  auto ret_found_keys = torch::tensor(found_keys);
  auto ret_missing_keys = torch::tensor(missing_keys);
  return std::make_tuple(data, ret_found_keys, ret_missing_keys);
}

std::tuple<torch::Tensor, torch::Tensor> FeatureCache2::Replace(
    torch::Tensor indices, torch::Tensor values) {
  // Update the tensor with the new values. just iterate over the indices.
  // randomly pick a palce from index_map_ and update the tensor_ with the
  // values.
  auto [updated_keys, skipped_keys] = key_cache_.Replace(indices);
  for (int64_t i = 0; i < updated_keys.size(); i++) {
    if (index_map_.find(updated_keys[i]) == index_map_.end()) {
      index_map_[updated_keys[i]] = index_map_.size();
    }
    tensor_[index_map_[updated_keys[i]]] = values[i];
  }
  return std::make_tuple(
      torch::tensor(updated_keys), torch::tensor(skipped_keys));
}
}  // namespace storage
}  // namespace graphbolt