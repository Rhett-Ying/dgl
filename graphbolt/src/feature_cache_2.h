#ifndef _GRAPHBOLT_FEATURE_CACHE_2_H_
#define _GRAPHBOLT_FEATURE_CACHE_2_H_

#include <torch/custom_class.h>
#include <torch/torch.h>

#include <random>
#include <unordered_map>
#include <vector>

namespace graphbolt {
namespace storage {

struct BaseCache {
  using KeysT = std::vector<int64_t>;
  using AddedPositionsT = std::unordered_map<int64_t, int64_t>;
  using EvictedKeysT = std::unordered_set<int64_t>;
  using ReplaceResultT = std::tuple<AddedPositionsT, EvictedKeysT>;
  virtual void Query(const KeysT& keys) = 0;
  virtual ReplaceResultT Replace(const KeysT& keys) = 0;
  virtual int64_t Capacity() = 0;
  virtual ~BaseCache() = default;
};

struct RandomIntGenerator {
  std::random_device rd;
  std::mt19937 gen;
  std::uniform_int_distribution<int64_t> dis;
  RandomIntGenerator(int64_t min, int64_t max) : gen(rd()), dis(min, max) {}
  int64_t operator()() { return dis(gen); }
};

struct RandomCache : public BaseCache {
  RandomCache(int capacity) : capacity_(capacity), gen_(0, capacity - 1) {
    data_.resize(capacity);
  }
  void Query(const KeysT& keys) override {
    // As we randomly replace the keys, we don't need to do anything here.
  }
  ReplaceResultT Replace(const KeysT& keys) override {
    AddedPositionsT added_positions;
    EvictedKeysT evicted_keys;
    for (const auto& key : keys) {
      if (keys_.find(key) == keys_.end()) {
        if (keys_.size() < capacity_) {
          const int64_t pos = keys_.size();
          data_[pos] = key;
          keys_.insert({key, pos});
          added_positions[key] = pos;
        } else {
          const int64_t pos = gen_();
          evicted_keys.insert(data_[pos]);
          keys_.erase(data_[pos]);
          data_[pos] = key;
          keys_[key] = pos;
          added_positions[key] = pos;
          if (evicted_keys.find(data_[pos]) != evicted_keys.end()) {
            evicted_keys.erase(data_[pos]);
          }
          if (added_positions.find(data_[pos]) != added_positions.end()) {
            added_positions.erase(data_[pos]);
          }
        }
      } else {
        // NO-OP
        ;
      }
    }
    return std::make_tuple(added_positions, evicted_keys);
  }
  int64_t Capacity() override { return capacity_; }
  int capacity_;
  std::vector<int64_t> data_;
  std::unordered_map<int64_t, int64_t> keys_;
  RandomIntGenerator gen_;
};

template <class CacheT>
class KeyCache {
 public:
  KeyCache(int capacity) : cache_(capacity) {
    keys_.reserve(capacity);
    std::cout << "KeyCache::capacity:" << capacity << std::endl;
  }
  std::tuple<
      std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>,
      std::vector<int64_t>, std::vector<int64_t>>
  Query(torch::Tensor keys) {
    cache_.Query(std::vector<int64_t>(
        keys.data_ptr<int64_t>(), keys.data_ptr<int64_t>() + keys.size(0)));
    // Iterate over the keys and check if the key is present in the cache.
    // return the keys that are found and not found.
    auto keys_ptr = keys.data_ptr<int64_t>();
    std::vector<int64_t> found_keys;
    std::vector<int64_t> positions;
    std::vector<int64_t> missing_keys;
    std::vector<int64_t> found_positions;
    std::vector<int64_t> missing_positions;
    for (int64_t i = 0; i < keys.size(0); i++) {
      if (keys_.find(keys_ptr[i]) != keys_.end()) {
        found_keys.push_back(keys_ptr[i]);
        positions.push_back(keys_[keys_ptr[i]]);
        found_positions.push_back(i);
      } else {
        missing_keys.push_back(keys_ptr[i]);
        missing_positions.push_back(i);
      }
    }
    return std::make_tuple(
        found_keys, positions, missing_keys, found_positions,
        missing_positions);
  }
  std::unordered_map<int64_t, int64_t> Replace(torch::Tensor indices) {
    auto&& [unique_indices, _x] = at::_unique(indices);
    const auto& [positions, evicted_keys] = cache_.Replace(std::vector<int64_t>(
        unique_indices.data_ptr<int64_t>(),
        unique_indices.data_ptr<int64_t>() + unique_indices.size(0)));
    for (const auto& key : evicted_keys) {
      keys_.erase(key);
    }
    for (const auto& [key, pos] : positions) {
      keys_[key] = pos;
    }
    return positions;
  }
  std::unordered_map<int64_t, int64_t> keys_;
  CacheT cache_;
};

class FeatureCache2 : public torch::CustomClassHolder {
 public:
  FeatureCache2(const std::vector<int64_t>& shape, torch::ScalarType dtype);

  // data, found_keys, missing_keys, found_positions, missing_positions.
  std::tuple<
      torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
  Query(torch::Tensor indices);

  // updated_keys, skipped_keys.
  std::tuple<torch::Tensor, torch::Tensor> Replace(
      torch::Tensor indices, torch::Tensor values);

  int64_t NumBytes() const { return tensor_.numel() * tensor_.element_size(); }

  static c10::intrusive_ptr<FeatureCache2> Create(
      const std::vector<int64_t>& shape, torch::ScalarType dtype) {
    return c10::make_intrusive<FeatureCache2>(shape, dtype);
  }

 private:
  torch::Tensor tensor_;
  KeyCache<RandomCache> key_cache_;
};
}  // namespace storage
}  // namespace graphbolt

#endif  // _GRAPHBOLT_FEATURE_CACHE_2_H_