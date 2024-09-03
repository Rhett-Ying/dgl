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
  std::vector<int64_t> Query(torch::Tensor keys) {
    // TODO
    return {};
  }
  int capacity_;
  std::unordered_map<int64_t, int64_t> index_map_;
};

class FeatureCache2 : public torch::CustomClassHolder {
 public:
  FeatureCache2(const std::vector<int64_t>& shape, torch::ScalarType dtype);

  std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Query(
      torch::Tensor indices);

  torch::Tensor Replace(torch::Tensor indices, torch::Tensor values);

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