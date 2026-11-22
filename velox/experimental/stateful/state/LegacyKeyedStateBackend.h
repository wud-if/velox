/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

// Legacy bridge: the pre-typed-API uint32_t-keyed state world, restored
// verbatim from the state before the operator-owned keyed state
// migration. Upstream's migration replaced the untyped
// StateBackend::createKeyedStateBackend() factory with a null return
// ("there is no factory-created heap backend") and turned the
// KeyedStateBackend base's state factories into NYIs on the heap
// backend, leaving not-yet-migrated operators without a state path.
// The windowjoin/window-aggregation operators still key state by the
// uint32_t key-group hash, so until they migrate to the typed
// KeyedStateSpec API this bridge restores that world behind
// HashMapStateBackend::createKeyedStateBackend(). Everything lives in
// the stateful::legacy namespace and nothing else includes it;
// drop this file when the operators adopt the typed API.

#include <climits>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "velox/common/base/BitUtil.h"
#include "velox/common/base/Exceptions.h"
#include "velox/experimental/stateful/InternalTimerService.h"
#include "velox/experimental/stateful/Triggerable.h"
#include "velox/experimental/stateful/state/KeyedStateBackend.h"
#include "velox/experimental/stateful/state/State.h"
#include "velox/experimental/stateful/state/StateDescriptor.h"
#include "velox/experimental/stateful/window/Window.h"

namespace facebook::velox::stateful::legacy {

// The pre-migration StateDescriptor carried the key-group bucket count
// (default 1024); the descriptor dropped it, so the legacy bridge pins it.
constexpr int32_t kLegacyKeyGroupNumber = 1024;

namespace {
int32_t roundUpToPowerOfTwo(int32_t x) {
  x = x - 1;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return x + 1;
}
} // namespace

template <typename K, typename N, typename S>
struct StateMapEntry {
  K key_;
  N namespace_;
  S state_;
  uint64_t hash_;
  std::shared_ptr<StateMapEntry<K, N, S>> next_;
  int32_t entryVersion_;
  int32_t stateVersion_;

  StateMapEntry(
      K key,
      N ns,
      S state,
      uint64_t hash,
      std::shared_ptr<StateMapEntry<K, N, S>>& next,
      int32_t entryVersion,
      int32_t stateVersion)
      : key_(key),
        namespace_(ns),
        state_(state),
        hash_(hash),
        next_(next),
        entryVersion_(entryVersion),
        stateVersion_(stateVersion) {}

  StateMapEntry(StateMapEntry<K, N, S> other, int entryVersion)
      : StateMapEntry(
            other.key_,
            other.namespace_,
            other.state_,
            other.hash_,
            other.next_,
            entryVersion,
            other.stateVersion_) {}

  bool operator==(const StateMapEntry<K, N, S>& entry) const {
    return entry.key_ == key_ && entry.namespace_ == namespace_ &&
        entry.state_ == state_;
  }

  uint64_t hashCode() {
    return bits::hashMix(bits::hashMix(key_, namespace_), state_);
  }
};

/**
 * This class is relevant to Flink org.apache.flink.runtime.state.heap.StateMap.
 * remove namespace first.
 * TODO: It is a simplified implementation, not equal to Flink.
 * @param <K> type of key
 * @param <N> type of namespace
 * @param <S> type of state
 */
template <typename K, typename N, typename S>
class StateMap {
#define MIN_TRANSFERRED_PER_INCREMENTAL_REHASH 4
#define MINIMUM_CAPACITY 4
#define MAXIMUM_CAPACITY 1 << 30
#define DEFAULT_CAPACITY 128
#define MAX_ARRAY_SIZE std::numeric_limits<int32_t>::max() - 8

 public:
  StateMap() : StateMap(DEFAULT_CAPACITY) {}
  StateMap(int32_t capacity) {
    primaryTable_ = empty_;
    incrementalRehashTable_ = empty_;
    highestRequiredSnapshotVersion_ = 0;
    stateMapVersion_ = 0;
    rehashIndex_ = 0;
    primaryTableSize_ = 0;
    incrementalRehashTableSize_ = 0;
    modCount_ = 0;
    lastNamespace_ = N{};

    if (capacity < 0) {
      threshold_ = -1;
      return;
    }

    if (capacity < MINIMUM_CAPACITY) {
      capacity = MINIMUM_CAPACITY;
    } else if (capacity > MAXIMUM_CAPACITY) {
      capacity = MAXIMUM_CAPACITY;
    } else {
      capacity = roundUpToPowerOfTwo(capacity);
    }

    primaryTable_ = makeTable(capacity);
  }

  S get(K key, N ns) {
    uint64_t hash = computeHashForOperationAndDoIncrementalRehash(key, ns);
    int32_t requiredVersion = highestRequiredSnapshotVersion_;
    std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& tab =
        selectActiveTable(hash);
    uint64_t index = hash & (tab.size() - 1);
    for (auto e = tab[index]; e != nullptr; e = e->next_) {
      if (e->hash_ == hash && e->key_ == key && e->namespace_ == ns) {
        if (e->stateVersion_ < requiredVersion) {
          if (e->entryVersion_ < requiredVersion) {
            e = handleChainedEntryCopyOnWrite(tab, hash & (tab.size() - 1), e);
          }
          e->stateVersion_ = stateMapVersion_;
          /// TODO: use type serializer to copy a state
          // e->state_ =
        }
        return e->state_;
      }
    }
    return nullptr;
  }

  void put(K key, N ns, S state) {
    std::shared_ptr<StateMapEntry<K, N, S>> e = putEntry(key, ns);
    e->state_ = state;
    e->stateVersion_ = stateMapVersion_;
  }

  bool containsKey(K key, N ns) {
    uint64_t hash = computeHashForOperationAndDoIncrementalRehash(key, ns);
    std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& tab =
        selectActiveTable(hash);
    uint64_t index = hash & (tab.size() - 1);
    for (auto e = tab[index]; e != nullptr; e = e->next_) {
      if (e->hash_ == hash && e->namespace_ == ns && e->key_ == key) {
        return true;
      }
    }
    return false;
  }

  void remove(K key, N ns) {
    removeEntry(key, ns);
  }

  size_t size() {
    return primaryTableSize_ + incrementalRehashTableSize_;
  }

 private:
  std::vector<std::shared_ptr<StateMapEntry<K, N, S>>> primaryTable_;
  std::vector<std::shared_ptr<StateMapEntry<K, N, S>>> incrementalRehashTable_;
  std::vector<std::shared_ptr<StateMapEntry<K, N, S>>> empty_;
  int32_t highestRequiredSnapshotVersion_{0};
  int32_t stateMapVersion_{0};
  // selectActiveTable() routes lookups using rehashIndex_; garbage here sends
  // get()/put() to the empty incremental table and causes tab[index] OOB.
  int32_t rehashIndex_{0};
  uint64_t primaryTableSize_{0};
  uint64_t incrementalRehashTableSize_{0};
  uint64_t modCount_{0};
  uint64_t threshold_{0};
  N lastNamespace_{};
  std::set<int32_t> snapshotVersions_;

  std::shared_ptr<StateMapEntry<K, N, S>> putEntry(K key, N ns) {
    uint64_t hash = computeHashForOperationAndDoIncrementalRehash(key, ns);
    std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& tab =
        selectActiveTable(hash);
    uint64_t index = hash & (tab.size() - 1);
    for (auto e = tab[index]; e != nullptr; e = e->next_) {
      if (e->hash_ == hash && e->key_ == key && e->namespace_ == ns) {
        if (e->entryVersion_ < highestRequiredSnapshotVersion_) {
          e = handleChainedEntryCopyOnWrite(tab, index, e);
        }
        return e;
      }
    }
    ++modCount_;
    if (size() > threshold_) {
      doubleCapacity();
    }
    return addNewStateMapEntry(tab, key, ns, hash);
  }

  std::shared_ptr<StateMapEntry<K, N, S>> removeEntry(K key, N ns) {
    uint64_t hash = computeHashForOperationAndDoIncrementalRehash(key, ns);
    std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& tab =
        selectActiveTable(hash);
    uint64_t index = hash & (tab.size() - 1);
    for (std::shared_ptr<StateMapEntry<K, N, S>> e = tab[index], prev = nullptr;
         e != nullptr;
         prev = e, e = e->next_) {
      if (e->hash_ == hash && e->key_ == key && e->namespace_ == ns) {
        if (prev == nullptr) {
          tab[index] = e->next_;
        } else {
          if (prev->entryVersion_ < highestRequiredSnapshotVersion_) {
            prev = handleChainedEntryCopyOnWrite(tab, index, prev);
          }
          prev->next_ = e->next_;
        }
        ++modCount_;
        if (&tab == &primaryTable_) {
          --primaryTableSize_;
        } else {
          --incrementalRehashTableSize_;
        }
        return e;
      }
    }
    return nullptr;
  }

  std::shared_ptr<StateMapEntry<K, N, S>> addNewStateMapEntry(
      std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& table,
      K key,
      N& ns,
      uint64_t hash) {
    if (ns == lastNamespace_) {
      ns = lastNamespace_;
    } else {
      lastNamespace_ = ns;
    }

    uint64_t index = hash & (table.size() - 1);
    std::shared_ptr<StateMapEntry<K, N, S>> newEntry =
        std::make_shared<StateMapEntry<K, N, S>>(
            key,
            ns,
            nullptr,
            hash,
            table[index],
            stateMapVersion_,
            stateMapVersion_);
    table[index] = newEntry;
    if (&table == &primaryTable_) {
      ++primaryTableSize_;
    } else {
      ++incrementalRehashTableSize_;
    }
    return newEntry;
  }

  std::shared_ptr<StateMapEntry<K, N, S>> handleChainedEntryCopyOnWrite(
      std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& tab,
      uint64_t mapIdx,
      std::shared_ptr<StateMapEntry<K, N, S>>& untilEntry) {
    int32_t required = highestRequiredSnapshotVersion_;
    std::shared_ptr<StateMapEntry<K, N, S>> current = tab[mapIdx];
    std::shared_ptr<StateMapEntry<K, N, S>> copy = nullptr;
    if (current->entryVersion_ < required) {
      copy =
          std::make_shared<StateMapEntry<K, N, S>>(*current, stateMapVersion_);
      tab[mapIdx] = copy;
    } else {
      copy = current;
    }

    while (current != untilEntry) {
      current = current->next_;
      if (current->entryVersion_ < required) {
        copy->next_ = std::make_shared<StateMapEntry<K, N, S>>(
            *current, stateMapVersion_);
        copy = copy->next_;
      } else {
        copy = current;
      }
    }
    return copy;
  }

  std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& selectActiveTable(
      uint64_t hashCode) {
    return (hashCode & (primaryTable_.size() - 1)) >= rehashIndex_
        ? primaryTable_
        : incrementalRehashTable_;
  }

  uint64_t compositeHash(K key, N ns) {
    if constexpr (std::is_same<N, TimeWindow>::value) {
      return bits::hashMix(key, ns.hashCode());
    } else {
      return bits::hashMix(key, ns);
    }
  }

  void incrementalRehash() {
    std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& oldMap =
        primaryTable_;
    std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& newMap =
        incrementalRehashTable_;

    size_t oldCapacity = oldMap.size();
    size_t newMask = newMap.size() - 1;
    int32_t requiredVersion = highestRequiredSnapshotVersion_;
    int32_t rhIdx = rehashIndex_;
    int32_t transferred = 0;

    while (transferred < MIN_TRANSFERRED_PER_INCREMENTAL_REHASH) {
      std::shared_ptr<StateMapEntry<K, N, S>> e = oldMap[rhIdx];
      while (e != nullptr) {
        if (e->entryVersion_ < requiredVersion) {
          e = std::make_shared<StateMapEntry<K, N, S>>(*e, stateMapVersion_);
        }
        std::shared_ptr<StateMapEntry<K, N, S>> n = e->next_;
        uint64_t pos = e->hash_ & newMask;
        e->next_ = newMap[pos];
        newMap[pos] = e;
        e = n;
        ++transferred;
      }

      oldMap[rhIdx] = nullptr;
      if (++rhIdx == oldCapacity) {
        primaryTable_ = newMap;
        incrementalRehashTable_ = empty_;
        primaryTableSize_ += incrementalRehashTableSize_;
        incrementalRehashTableSize_ = 0;
        rehashIndex_ = 0;
        return;
      }
    }

    primaryTableSize_ -= transferred;
    incrementalRehashTableSize_ += transferred;
    rehashIndex_ = rhIdx;
  }

  bool isRehashing() {
    return empty_ != incrementalRehashTable_;
  }

  uint64_t computeHashForOperationAndDoIncrementalRehash(K key, N ns) {
    if (isRehashing()) {
      incrementalRehash();
    }
    return compositeHash(key, ns);
  }

  void doubleCapacity() {
    VELOX_CHECK(!isRehashing(), "There is already a rehash in progress.");
    std::vector<std::shared_ptr<StateMapEntry<K, N, S>>>& oldMap =
        primaryTable_;
    int oldCapacity = oldMap.size();
    if (oldCapacity == MAXIMUM_CAPACITY) {
      return;
    }
    rehashIndex_ = 0;
    incrementalRehashTable_ = makeTable(oldCapacity * 2);
  }

  std::vector<std::shared_ptr<StateMapEntry<K, N, S>>> makeTable(
      uint64_t newCapacity) {
    if (newCapacity < MAXIMUM_CAPACITY) {
      threshold_ = (newCapacity >> 1) + (newCapacity >> 2);
    } else {
      if (size() > MAX_ARRAY_SIZE) {
        VELOX_FAIL(
            "Maximum capacity of CopyOnWriteStateMap is reached and the job cannot continue.");
      } else {
        LOG(WARNING)
            << "Maximum capacity of 2^30 in StateMap reached. Cannot increase hash map size.";
        threshold_ = MAX_ARRAY_SIZE;
      }
    }
    std::vector<std::shared_ptr<StateMapEntry<K, N, S>>> newMap;
    newMap.resize(newCapacity);
    return newMap;
  }
};

/**
 * This class is relevant to Flink
 * org.apache.flink.runtime.state.heap.StateTable. remove namespace first. And
 * implement it here instead of making it as a interface.
 * @param <K> type of key
 * @param <S> type of state
 */
template <typename K, typename N, typename S>
class StateTable {
 public:
  StateTable(int keyGroupNumber)
      : keyGroupedStates_(keyGroupNumber), keyGroupNumber_(keyGroupNumber) {}

  S get(const K& key, const N& ns) {
    if (keyGroupNumber_ == 0) {
      return nullptr;
    }
    int keyGroupIndex = std::hash<K>()(key) % keyGroupNumber_;
    return keyGroupedStates_[keyGroupIndex].get(key, ns);
  }

  void put(const K& key, const N& ns, const S& state) {
    if (keyGroupNumber_ == 0) {
      return;
    }
    int keyGroupIndex = std::hash<K>()(key) % keyGroupNumber_;
    keyGroupedStates_[keyGroupIndex].put(key, ns, state);
  }

  void remove(const K& key, const N& ns) {
    if (keyGroupNumber_ == 0) {
      return;
    }
    int keyGroupIndex = std::hash<K>()(key) % keyGroupNumber_;
    keyGroupedStates_[keyGroupIndex].remove(key, ns);
  }

  void clear() {
    keyGroupedStates_.assign(keyGroupNumber_, StateMap<K, N, S>{});
  }

 private:
  std::vector<StateMap<K, N, S>> keyGroupedStates_;
  int keyGroupNumber_;
};

// This class is relevant to Flink HeapMapState.
template <typename K, typename N, typename V>
class HeapListState : public ListState<K, N, V> {
 public:
  HeapListState(int keyGroupNumber) {
    stateTable_ =
        std::make_unique<StateTable<K, N, std::shared_ptr<std::vector<V>>>>(
            keyGroupNumber);
  }

  std::vector<V> get(const K& key, const N& ns) override {
    auto currentList = stateTable_->get(key, ns);
    return currentList == nullptr ? std::vector<V>{} : *currentList;
  }

  void add(const K& key, const N& ns, const V& value) override {
    auto currentList = getOrCreate(key, ns);
    currentList->push_back(value);
  }

  /**
  void addAll(int keyGroupIndex, std::list<T>& values) override {
    std::shared_ptr<std::list<T>> currentList = getOrCreate(keyGroupIndex);
    currentList->insert(currentList->end(), values.begin(), values.end());
  }

  void update(int keyGroupIndex, std::list<T>& values) override {
    std::shared_ptr<std::list<T>> currentList = getOrCreate(keyGroupIndex);
    *currentList = values; // Replace the current list with the new values
  }
  */

  void clear() override {
    stateTable_->clear();
  }

  void remove(const K& key, const N& ns) override {
    stateTable_->remove(key, ns);
  }

 private:
  std::shared_ptr<std::vector<V>> getOrCreate(const K& key, const N& ns) {
    std::shared_ptr<std::vector<V>> currentList = stateTable_->get(key, ns);
    if (currentList == nullptr) {
      currentList = std::make_shared<std::vector<V>>();
      stateTable_->put(key, ns, currentList);
    }
    return currentList;
  }

  std::unique_ptr<StateTable<K, N, std::shared_ptr<std::vector<V>>>>
      stateTable_;
};

// This class is relevant to Flink HeapMapState.
template <typename K, typename N, typename V>
class HeapValueState : public ValueState<K, N, V> {
 public:
  HeapValueState(int keyGroupNumber) {
    VELOX_CHECK(keyGroupNumber > 0, "keyGroupNumber must be greater than 0");
    stateTable_ = std::make_unique<StateTable<K, N, V>>(keyGroupNumber);
  }

  V value(const K& key, const N& ns) override {
    return stateTable_->get(key, ns);
  }

  void update(const K& key, const N& ns, const V& value) override {
    stateTable_->put(key, ns, value);
  }

  void remove(const K& key, const N& ns) override {
    stateTable_->remove(key, ns);
  }

  void clear() override {
    stateTable_->clear();
  }

 private:
  std::unique_ptr<StateTable<K, N, V>> stateTable_;
};

// This class is relevant to Flink HeapMapState.
template <typename K, typename N, typename UK, typename UV>
class HeapMapState : public MapState<K, N, UK, UV> {
 public:
  HeapMapState(int keyGroupNumber) : keyGroupNumber_(keyGroupNumber) {
    stateTable_ =
        std::make_unique<StateTable<K, N, std::shared_ptr<std::map<UK, UV>>>>(
            keyGroupNumber);
  }

  std::optional<UV> get(const K& key, const N& ns, const UK& userKey) override {
    std::shared_ptr<std::map<UK, UV>> currentMap = getOrCreate(key, ns);
    return currentMap->count(userKey)
        ? std::optional<UV>((*currentMap)[userKey])
        : std::nullopt;
  }

  void put(const K& key, const N& ns, const UK& userKey, const UV& value)
      override {
    std::shared_ptr<std::map<UK, UV>> currentMap = getOrCreate(key, ns);
    currentMap->insert({userKey, value});
  }

  std::map<UK, UV> entries(const K& key, const N& ns) override {
    return *getOrCreate(key, ns).get();
  }

  void clear() override {
    stateTable_->clear();
  }

  void remove(const K& key, const N& ns, const UK& userKey) override {
    std::shared_ptr<std::map<UK, UV>> currentMap = getOrCreate(key, ns);
    currentMap->erase(userKey);
  }

 private:
  std::shared_ptr<std::map<UK, UV>> getOrCreate(const K& key, const N& ns) {
    std::shared_ptr<std::map<UK, UV>> currentMap = stateTable_->get(key, ns);
    if (currentMap == nullptr) {
      currentMap = std::make_shared<std::map<UK, UV>>();
      stateTable_->put(key, ns, currentMap);
    }
    return currentMap;
  }

  std::unique_ptr<StateTable<K, N, std::shared_ptr<std::map<UK, UV>>>>
      stateTable_;
  int keyGroupNumber_;
};

// This class is relevant to Flink HeapKeyedStateBackend.
class HeapKeyedStateBackend : public KeyedStateBackend {
 public:
  // The byte-level checkpoint plane is the typed backends' domain; the
  // legacy bridge predates it and its operators do not take part in the
  // typed checkpoint walk. Fail loud if the walk ever reaches a legacy
  // backend instead of silently dropping state.
  std::string snapshotBytes() override {
    VELOX_UNSUPPORTED(
        "LegacyKeyedStateBackend does not support byte-level checkpointing");
  }

  void restoreBytes(const std::string& /*bytes*/) override {
    VELOX_UNSUPPORTED(
        "LegacyKeyedStateBackend does not support byte-level checkpointing");
  }

  // TODO: use template to support different key type.
  std::shared_ptr<MapState<uint32_t, int, RowVectorPtr, int>>
  getOrCreateMapState(StateDescriptor& stateDescriptor) override;

  std::shared_ptr<ListState<uint32_t, int64_t, RowVectorPtr>>
  getOrCreateListState(StateDescriptor& stateDescriptor) override;

  std::shared_ptr<ValueState<int64_t, int64_t, RowVectorPtr>>
  getOrCreateValueState(StateDescriptor& stateDescriptor) override;

  std::shared_ptr<InternalTimerService<int64_t, int64_t>> createTimerService(
      Triggerable<int64_t, int64_t>* triggerable) override;

  std::shared_ptr<ValueState<uint32_t, TimeWindow, RowVectorPtr>>
  getOrCreateGroupValueState(StateDescriptor& stateDescriptor) override;

  std::shared_ptr<MapState<uint32_t, int, TimeWindow, TimeWindow>>
  getOrCreateGroupMapState(StateDescriptor& stateDescriptor) override;

  std::shared_ptr<MapState<uint32_t, int, uint32_t, RowVectorPtr>>
  getOrCreateRankMapState(StateDescriptor& stateDescriptor) override;

  virtual std::shared_ptr<InternalTimerService<int64_t, TimeWindow>>
  createGroupWindowAggTimerService(
      Triggerable<int64_t, TimeWindow>* triggerable) override;

  void snapshot(
      int64_t checkpointId,
      int64_t timestamp,
      CheckpointOptions checkpointOptions) override;

  void notifyCheckpointComplete(int64_t checkpointId) override;

  void notifyCheckpointAborted(int64_t checkpointId) override;

 private:
  std::map<std::string, StatePtr> keyValueStatesByName_;
};

std::shared_ptr<MapState<uint32_t, int, RowVectorPtr, int>>
HeapKeyedStateBackend::getOrCreateMapState(StateDescriptor& stateDescriptor) {
  auto stateIte = keyValueStatesByName_.find(stateDescriptor.name());
  if (stateIte != keyValueStatesByName_.end()) {
    return std::dynamic_pointer_cast<
        MapState<uint32_t, int, RowVectorPtr, int>>(stateIte->second);
  }
  auto state = std::make_shared<HeapMapState<uint32_t, int, RowVectorPtr, int>>(
      kLegacyKeyGroupNumber);
  keyValueStatesByName_.insert({stateDescriptor.name(), state});
  return state;
}

std::shared_ptr<ListState<uint32_t, int64_t, RowVectorPtr>>
HeapKeyedStateBackend::getOrCreateListState(StateDescriptor& stateDescriptor) {
  auto stateIte = keyValueStatesByName_.find(stateDescriptor.name());
  if (stateIte != keyValueStatesByName_.end()) {
    return std::dynamic_pointer_cast<
        ListState<uint32_t, int64_t, RowVectorPtr>>(stateIte->second);
  }
  auto state = std::make_shared<HeapListState<uint32_t, int64_t, RowVectorPtr>>(
      kLegacyKeyGroupNumber);
  keyValueStatesByName_.insert({stateDescriptor.name(), state});
  return state;
}

std::shared_ptr<ValueState<int64_t, int64_t, RowVectorPtr>>
HeapKeyedStateBackend::getOrCreateValueState(StateDescriptor& stateDescriptor) {
  auto stateIte = keyValueStatesByName_.find(stateDescriptor.name());
  if (stateIte != keyValueStatesByName_.end()) {
    return std::dynamic_pointer_cast<
        ValueState<int64_t, int64_t, RowVectorPtr>>(stateIte->second);
  }
  auto state = std::make_shared<HeapValueState<int64_t, int64_t, RowVectorPtr>>(
      kLegacyKeyGroupNumber);
  keyValueStatesByName_.insert({stateDescriptor.name(), state});
  return state;
}

std::shared_ptr<ValueState<uint32_t, TimeWindow, RowVectorPtr>>
HeapKeyedStateBackend::getOrCreateGroupValueState(
    StateDescriptor& stateDescriptor) {
  auto stateIte = keyValueStatesByName_.find(stateDescriptor.name());
  if (stateIte != keyValueStatesByName_.end()) {
    return std::dynamic_pointer_cast<
        ValueState<uint32_t, TimeWindow, RowVectorPtr>>(stateIte->second);
  }
  auto state =
      std::make_shared<HeapValueState<uint32_t, TimeWindow, RowVectorPtr>>(
          kLegacyKeyGroupNumber);
  keyValueStatesByName_.insert({stateDescriptor.name(), state});
  return state;
}

std::shared_ptr<MapState<uint32_t, int, TimeWindow, TimeWindow>>
HeapKeyedStateBackend::getOrCreateGroupMapState(
    StateDescriptor& stateDescriptor) {
  auto stateIte = keyValueStatesByName_.find(stateDescriptor.name());
  if (stateIte != keyValueStatesByName_.end()) {
    return std::dynamic_pointer_cast<
        MapState<uint32_t, int, TimeWindow, TimeWindow>>(stateIte->second);
  }
  auto state =
      std::make_shared<HeapMapState<uint32_t, int, TimeWindow, TimeWindow>>(
          kLegacyKeyGroupNumber);
  keyValueStatesByName_.insert({stateDescriptor.name(), state});
  return state;
}

std::shared_ptr<MapState<uint32_t, int, uint32_t, RowVectorPtr>>
HeapKeyedStateBackend::getOrCreateRankMapState(
    StateDescriptor& stateDescriptor) {
  auto stateIte = keyValueStatesByName_.find(stateDescriptor.name());
  if (stateIte != keyValueStatesByName_.end()) {
    return std::dynamic_pointer_cast<
        MapState<uint32_t, int, uint32_t, RowVectorPtr>>(stateIte->second);
  }
  auto state =
      std::make_shared<HeapMapState<uint32_t, int, uint32_t, RowVectorPtr>>(
          kLegacyKeyGroupNumber);
  keyValueStatesByName_.insert({stateDescriptor.name(), state});
  return state;
}

std::shared_ptr<InternalTimerService<int64_t, int64_t>>
HeapKeyedStateBackend::createTimerService(
    Triggerable<int64_t, int64_t>* triggerable) {
  return std::make_shared<InternalTimerService<int64_t, int64_t>>(triggerable);
}

std::shared_ptr<InternalTimerService<int64_t, TimeWindow>>
HeapKeyedStateBackend::createGroupWindowAggTimerService(
    Triggerable<int64_t, TimeWindow>* triggerable) {
  return std::make_shared<InternalTimerService<int64_t, TimeWindow>>(
      triggerable);
}

void HeapKeyedStateBackend::snapshot(
    int64_t checkpointId,
    int64_t timestamp,
    CheckpointOptions checkpointOptions) {
  // TODO: implement snapshot logic.
}

void HeapKeyedStateBackend::notifyCheckpointComplete(int64_t checkpointId) {
  // TODO: implement checkpoint complete logic.
}

void HeapKeyedStateBackend::notifyCheckpointAborted(int64_t checkpointId) {
  // TODO: implement checkpoint abort logic.
}

} // namespace facebook::velox::stateful::legacy
