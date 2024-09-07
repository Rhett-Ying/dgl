#ifndef _GRAPHBOLT_FEATURE_CACHE_2_H_
#define _GRAPHBOLT_FEATURE_CACHE_2_H_

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <random>
#include <unordered_map>
#include <vector>

namespace graphbolt {
namespace storage {

struct RandomIntGenerator {
  std::random_device rd;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> dis;
  RandomIntGenerator(int64_t min, int64_t max) : gen(rd()), dis(min, max) {}
  int64_t operator()() { return dis(gen); }
};

class KeyCache {
 public:
  KeyCache(int capacity) : data_(capacity, 0), gen_(0, capacity - 1) {
    capacity_ = capacity;
  }
  int64_t Query(int64_t key) {
    // TODO
    return 0;
  }
  std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>>
  Query(torch::Tensor keys) {
    // Iterate over the keys and check if the key is present in the cache.
    // return the keys that are found and not found.
    auto keys_ptr = keys.data_ptr<int64_t>();
    std::vector<int64_t> found_keys;
    std::vector<int64_t> positions;
    std::vector<int64_t> missing_keys;
    for (int64_t i = 0; i < keys.size(0); i++) {
      if (keys_.find(keys_ptr[i]) != keys_.end()) {
        found_keys.push_back(keys_ptr[i]);
        positions.push_back(keys_[keys_ptr[i]]);
      } else {
        missing_keys.push_back(keys_ptr[i]);
      }
    }
    return std::make_tuple(found_keys, positions, missing_keys);
  }
  std::unordered_map<int64_t, int64_t> Replace(torch::Tensor indices) {
    std::unordered_map<int64_t, int64_t> positions;
    auto indices_ptr = indices.data_ptr<int64_t>();
    for (int64_t i = 0; i < indices.size(0); i++) {
      if (keys_.find(indices_ptr[i]) == keys_.end()) {
        if (keys_.size() < capacity_) {
          data_[keys_.size()] = indices_ptr[i];
          int64_t pos = keys_.size();
          keys_.insert({indices_ptr[i], pos});
          positions[indices_ptr[i]] = pos;
        } else {
          auto pos = gen_();
          keys_.erase(data_[pos]);
          data_[pos] = indices_ptr[i];
          keys_[indices_ptr[i]] = pos;
          positions[indices_ptr[i]] = pos;
        }
        // print keys_;
        for (auto const& x : keys_) {
          std::cout << x.first << ": " << x.second << std::endl;
        }
      } else {
        // NO-OP
        ;
        // positions[indices_ptr[i]] = keys_[indices_ptr[i]];
      }
    }

    return positions;
  }
  int capacity_;
  std::vector<int64_t> data_;
  std::unordered_map<int64_t, int64_t> keys_;
  RandomIntGenerator gen_;
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