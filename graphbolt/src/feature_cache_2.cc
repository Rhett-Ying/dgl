#include "./feature_cache_2.h"

namespace graphbolt {
namespace storage {

FeatureCache2::FeatureCache2(
    const std::vector<int64_t>& shape, torch::ScalarType dtype)
    : key_cache_(shape[0]) {
  tensor_ = torch::empty(shape, torch::TensorOptions().dtype(dtype));
}

std::tuple<
    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
FeatureCache2::Query(torch::Tensor indices) {
  // Retrieve the found keys and missing keys from key_cache_.
  auto
      [found_keys, positions, missing_keys, found_positions,
       missing_positions] = key_cache_.Query(indices);

  // read data from tensor_.
  auto data =
      torch::empty({found_keys.size(), tensor_.size(1)}, tensor_.options());
  for (int64_t i = 0; i < positions.size(); i++) {
    data[i] = tensor_[positions[i]];
  }

  // return found_keys, missing_keys as tensor.
  auto ret_found_keys = torch::tensor(found_keys);
  auto ret_missing_keys = torch::tensor(missing_keys);
  auto ret_found_positions = torch::tensor(found_positions);
  auto ret_missing_positions = torch::tensor(missing_positions);
  return std::make_tuple(
      data, ret_found_keys, ret_missing_keys, ret_found_positions,
      ret_missing_positions);
}

std::tuple<torch::Tensor, torch::Tensor> FeatureCache2::Replace(
    torch::Tensor indices, torch::Tensor values) {
  auto positions = key_cache_.Replace(indices);
  std::vector<int64_t> updated_keys;
  std::vector<int64_t> skipped_keys;
  auto indices_ptr = indices.data_ptr<int64_t>();
  for (int64_t i = 0; i < indices.size(0); i++) {
    auto iter = positions.find(indices_ptr[i]);
    if (iter == positions.end()) {
      skipped_keys.push_back(indices_ptr[i]);
      continue;
    }
    tensor_[iter->second] = values[i];
    updated_keys.push_back(indices_ptr[i]);
  }

  return std::make_tuple(
      torch::tensor(updated_keys), torch::tensor(skipped_keys));
}
}  // namespace storage
}  // namespace graphbolt