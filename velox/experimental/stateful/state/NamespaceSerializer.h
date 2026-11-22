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

#include <string>
#include "velox/experimental/stateful/TypeSerializer.h"
#include "velox/experimental/stateful/state/Namespace.h"

namespace facebook::velox::stateful {

/// Serializes VoidNamespace. The namespace is stateless: any payload is
/// empty and all instances are equal.
class VoidNamespaceSerializer : public TypeSerializer<VoidNamespace> {
 public:
  std::string schemaId() const override {
    return "void";
  }

  std::string serialize(const VoidNamespace& /* ns */) override {
    return "";
  }

  VoidNamespace deserialize(std::string_view /* str */) override {
    return VoidNamespace{};
  }
};

/// Serializes WindowEndNamespace: the payload is the 8-byte little-endian
/// window end, mirroring Flink's Long namespace serializer.
class WindowEndNamespaceSerializer : public TypeSerializer<WindowEndNamespace> {
 public:
  std::string schemaId() const override {
    return "window-end:i64";
  }

  std::string serialize(const WindowEndNamespace& ns) override {
    std::string payload;
    payload.reserve(sizeof(int64_t));
    for (size_t i = 0; i < sizeof(int64_t); ++i) {
      payload.push_back(static_cast<char>((ns.windowEnd() >> (8 * i)) & 0xFF));
    }
    return payload;
  }

  WindowEndNamespace deserialize(std::string_view str) override {
    VELOX_CHECK_EQ(
        str.size(),
        sizeof(int64_t),
        "Corrupt window-end namespace payload: expected 8 bytes");
    int64_t windowEnd = 0;
    for (size_t i = 0; i < sizeof(int64_t); ++i) {
      windowEnd |= static_cast<int64_t>(static_cast<uint8_t>(str[i]))
          << (8 * i);
    }
    return WindowEndNamespace(windowEnd);
  }
};

/// Compile-time binding of a namespace type to its serializer, in the
/// shape of TypeTraits: the primary template is undefined, so a namespace
/// type without a specialization fails to compile where a handle
/// constructs its serializer; a new namespace type plugs in here.
template <typename N>
struct NamespaceSerializerTraits;

template <>
struct NamespaceSerializerTraits<VoidNamespace> {
  static std::shared_ptr<TypeSerializer<VoidNamespace>> create() {
    return std::make_shared<VoidNamespaceSerializer>();
  }
};

template <>
struct NamespaceSerializerTraits<WindowEndNamespace> {
  static std::shared_ptr<TypeSerializer<WindowEndNamespace>> create() {
    return std::make_shared<WindowEndNamespaceSerializer>();
  }
};

} // namespace facebook::velox::stateful
