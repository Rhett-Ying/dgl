#ifndef _GRAPHBOLT_FEATURE_CACHE_2_H_
#define _GRAPHBOLT_FEATURE_CACHE_2_H_

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <unordered_map>
#include <vector>

namespace graphbolt {
namespace storage {

class KeyCache {
 public:
  KeyCache(int capacity) { capacity_ = capacity; }
  int64_t Query(int64_t key) {
    // TODO
    return 0;
  }
  std::tuple<std::vector<int64_t>, std::vector<int64_t>> Query(
      torch::Tensor keys) {
    // Iterate over the keys and check if the key is present in the cache.
    // return the keys that are found and not found.
    auto keys_ptr = keys.data_ptr<int64_t>();
    std::vector<int64_t> found_keys;
    std::vector<int64_t> missing_keys;
    for (int64_t i = 0; i < keys.size(0); i++) {
      if (keys_.find(keys_ptr[i]) != keys_.end()) {
        found_keys.push_back(keys_ptr[i]);
      } else {
        missing_keys.push_back(keys_ptr[i]);
      }
    }
    return std::make_tuple(found_keys, missing_keys);
  }
  std::tuple<std::vector<int64_t>, std::vector<int64_t>> Replace(
      torch::Tensor keys) {
    // Iterate over the keys and check if the key is present in the cache.
    // return the keys that are updated and skipped.
    auto keys_ptr = keys.data_ptr<int64_t>();
    std::vector<int64_t> updated_keys;
    std::vector<int64_t> skipped_keys;
    for (int64_t i = 0; i < keys.size(0); i++) {
      if (keys_.find(keys_ptr[i]) == keys_.end()) {
        updated_keys.push_back(keys_ptr[i]);
      } else {
        skipped_keys.push_back(keys_ptr[i]);
      }
    }
    // insert updated_keys into keys_.
    for (auto key : updated_keys) {
      keys_.insert(key);
    }
    return std::make_tuple(updated_keys, skipped_keys);
  }
  int capacity_;
  std::unordered_set<int64_t> keys_;
};

class FeatureCache2 : public torch::CustomClassHolder {
 public:
  FeatureCache2(const std::vector<int64_t>& shape, torch::ScalarType dtype);

  // data, found_keys, missing_keys.
  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor indices);

  // updated_keys, skipped_keys.
  std::tuple<torch::Tensor, torch::Tensor> Replace(
      torch::Tensor indices, torch::Tensor values);

  static c10::intrusive_ptr<FeatureCache2> Create(
      const std::vector<int64_t>& shape, torch::ScalarType dtype) {
    return c10::make_intrusive<FeatureCache2>(shape, dtype);
  }

 private:
  torch::Tensor tensor_;
  KeyCache key_cache_;
  std::unordered_map<int64_t, int64_t> index_map_;
};
}  // namespace storage
}  // namespace graphbolt

#endif  // _GRAPHBOLT_FEATURE_CACHE_2_H_