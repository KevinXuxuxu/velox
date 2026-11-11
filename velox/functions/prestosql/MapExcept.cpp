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

#include "velox/functions/prestosql/MapExcept.h"
#include <folly/container/F14Set.h>
#include "velox/expression/SimpleFunctionRegistry.h"
#include "velox/expression/VectorFunction.h"
#include "velox/functions/Registerer.h"
#include "velox/vector/FlatMapVector.h"

namespace facebook::velox::functions {
namespace {

// Looks up the flat map channel of each excluded key, through the scalar key
// index where the key type has one.
class ChannelFinder {
 public:
  ChannelFinder(const FlatMapVector& flatMap, const VectorPtr& keyElements)
      : flatMap_{flatMap},
        keyKind_{flatMap.keyType()->kind()},
        keyElements_{keyElements} {
    decodedElements_.decode(*keyElements_);
  }

  bool isNullAt(vector_size_t index) const {
    return decodedElements_.isNullAt(index);
  }

  std::optional<column_index_t> operator()(vector_size_t index) const {
    if (keyKind_ == TypeKind::INTEGER) {
      return flatMap_.getKeyChannel(decodedElements_.valueAt<int32_t>(index));
    }
    if (keyKind_ == TypeKind::BIGINT) {
      return flatMap_.getKeyChannel(decodedElements_.valueAt<int64_t>(index));
    }
    if (keyKind_ == TypeKind::VARCHAR) {
      return flatMap_.getKeyChannel(
          decodedElements_.valueAt<StringView>(index));
    }
    return flatMap_.getKeyChannel(keyElements_, index);
  }

 private:
  const FlatMapVector& flatMap_;
  const TypeKind keyKind_;
  const VectorPtr& keyElements_;
  DecodedVector decodedElements_;
};

// Copies 'decoded's row mapping for 'rows', leaving every other row on the zero
// allocateIndices() starts from.
BufferPtr copyIndices(
    const DecodedVector& decoded,
    const SelectivityVector& rows,
    memory::MemoryPool* pool) {
  auto indices = allocateIndices(decoded.size(), pool);
  auto* rawIndices = indices->asMutable<vector_size_t>();
  rows.applyToSelected(
      [&](vector_size_t row) { rawIndices[row] = decoded.index(row); });
  return indices;
}

// True when every selected row reads the flat map row of the same index, as
// for an unwrapped flat map or a dictionary that only adds nulls. The flat
// map's in-map buffers can then be shared.
bool sharesRowSpace(
    const DecodedVector& decodedMap,
    const SelectivityVector& rows) {
  if (decodedMap.isIdentityMapping()) {
    return true;
  }
  if (decodedMap.isConstantMapping()) {
    return false;
  }
  return rows.testSelected(
      [&](vector_size_t row) { return decodedMap.index(row) == row; });
}

// Handles a constant list of excluded keys over a flat map in its own row
// space. Every row drops the same channels, so the result keeps the input's map
// values and in-map buffers for the others and drops the excluded channels
// entirely.
void applyFlatMap(
    const VectorPtr& map,
    const VectorPtr& keys,
    const FlatMapVector& flatMap,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  // Row 0 describes the whole batch, but it has to be resolved through the
  // constant wrapping: indexing the base vector directly reads the wrong row.
  const auto& keysArray = *keys->wrappedVector()->asUnchecked<ArrayVector>();
  const auto keysIndex = keys->wrappedIndex(0);
  const auto begin = keysArray.offsetAt(keysIndex);
  const auto end = begin + keysArray.sizeAt(keysIndex);
  const ChannelFinder findChannel(flatMap, keysArray.elements());

  folly::F14FastSet<column_index_t> excluded;
  for (auto i = begin; i < end; ++i) {
    if (findChannel.isNullAt(i)) {
      continue;
    }
    if (const auto channel = findChannel(i)) {
      excluded.insert(*channel);
    }
  }

  if (excluded.empty()) {
    context.moveOrCopyResult(map, rows, result);
    return;
  }

  const auto numDistinctKeys = flatMap.numDistinctKeys();
  const auto numKept =
      numDistinctKeys - static_cast<vector_size_t>(excluded.size());
  std::vector<BaseVector::CopyRange> keyRanges;
  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;
  keyRanges.reserve(numKept);
  mapValues.reserve(numKept);
  inMaps.reserve(numKept);
  for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
    if (excluded.contains(channel)) {
      continue;
    }
    keyRanges.push_back(
        {static_cast<vector_size_t>(channel),
         static_cast<vector_size_t>(mapValues.size()),
         1});
    mapValues.push_back(flatMap.mapValuesAt(channel));
    // A missing in-map buffer means the key is present in every row, the same
    // convention FlatMapVector::isInMap() follows.
    if (channel < flatMap.inMaps().size()) {
      inMaps.push_back(flatMap.inMaps()[channel]);
    } else {
      inMaps.emplace_back(nullptr);
    }
  }

  VectorPtr distinctKeys;
  if (numKept > 0) {
    distinctKeys =
        BaseVector::create(flatMap.keyType(), numKept, context.pool());
    distinctKeys->copyRanges(flatMap.distinctKeys().get(), keyRanges);
  }

  auto localResult = std::make_shared<FlatMapVector>(
      context.pool(),
      outputType,
      flatMap.nulls(),
      flatMap.size(),
      std::move(distinctKeys),
      std::move(mapValues),
      std::move(inMaps));

  context.moveOrCopyResult(localResult, rows, result);
}

// Handles per-row excluded keys, a wrapped flat map, or both. Finding what to
// remove visits only the (channel, row) pairs an excluded key hits. In the flat
// map's row space, only the in-map buffers of those channels are copied; a
// wrapped flat map has every in-map buffer rebuilt in the output row space.
void applyFlatMap(
    const VectorPtr& map,
    const DecodedVector& decodedMap,
    bool sameRows,
    const DecodedVector& decodedKeys,
    const FlatMapVector& flatMap,
    const SelectivityVector& rows,
    const TypePtr& outputType,
    exec::EvalCtx& context,
    VectorPtr& result) {
  const auto numDistinctKeys = flatMap.numDistinctKeys();
  // In the flat map's row space, in-map buffers stay shareable. Otherwise they
  // are rebuilt in the output row space.
  const auto numRows = sameRows ? flatMap.size() : decodedMap.size();
  const auto& keysArray = *decodedKeys.base()->asUnchecked<ArrayVector>();
  const ChannelFinder findChannel(flatMap, keysArray.elements());

  // In the flat map's row space, the in-map buffer of a channel that loses a
  // key is copied once and its bits are cleared in place. Otherwise, the rows
  // that lose the key are recorded and applied while rebuilding.
  std::vector<BufferPtr> changed(numDistinctKeys);
  bool anyRemoved = false;
  rows.applyToSelected([&](vector_size_t row) {
    const auto mapRow = decodedMap.index(row);
    const auto keysIndex = decodedKeys.index(row);
    const auto begin = keysArray.offsetAt(keysIndex);
    const auto end = begin + keysArray.sizeAt(keysIndex);
    for (auto i = begin; i < end; ++i) {
      if (findChannel.isNullAt(i)) {
        continue;
      }
      const auto channel = findChannel(i);
      if (!channel.has_value() || !flatMap.isInMap(*channel, mapRow)) {
        continue;
      }
      auto& bits = changed[*channel];
      if (bits == nullptr) {
        if (!sameRows) {
          bits = AlignedBuffer::allocate<bool>(numRows, context.pool(), false);
        } else if (
            *channel < flatMap.inMaps().size() &&
            flatMap.inMaps()[*channel] != nullptr) {
          bits =
              AlignedBuffer::copy(context.pool(), flatMap.inMaps()[*channel]);
        } else {
          bits = AlignedBuffer::allocate<bool>(numRows, context.pool(), true);
        }
      }
      if (sameRows) {
        bits::clearBit(bits->asMutable<uint64_t>(), row);
      } else {
        bits::setBit(bits->asMutable<uint64_t>(), row);
      }
      anyRemoved = true;
    }
  });

  if (!anyRemoved) {
    context.moveOrCopyResult(map, rows, result);
    return;
  }

  std::vector<VectorPtr> mapValues;
  std::vector<BufferPtr> inMaps;
  BufferPtr nulls;
  if (sameRows) {
    mapValues = flatMap.mapValues();
    inMaps = flatMap.inMaps();
    inMaps.resize(numDistinctKeys);
    for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
      if (changed[channel] != nullptr) {
        inMaps[channel] = std::move(changed[channel]);
      }
    }
    nulls = flatMap.nulls();
  } else {
    const auto indices = copyIndices(decodedMap, rows, context.pool());
    mapValues.reserve(numDistinctKeys);
    inMaps.reserve(numDistinctKeys);
    for (column_index_t channel = 0; channel < numDistinctKeys; ++channel) {
      const auto& values = flatMap.mapValuesAt(channel);
      mapValues.push_back(
          values == nullptr
              ? nullptr
              : BaseVector::wrapInDictionary(
                    BufferPtr(nullptr), indices, numRows, values));

      const auto* baseInMap = flatMap.rawInMapsAt(channel);
      const auto* rawRemoved = changed[channel] != nullptr
          ? changed[channel]->as<uint64_t>()
          : nullptr;
      if (baseInMap == nullptr && rawRemoved == nullptr) {
        inMaps.emplace_back(nullptr);
        continue;
      }
      auto inMap =
          AlignedBuffer::allocate<bool>(numRows, context.pool(), false);
      auto* rawInMap = inMap->asMutable<uint64_t>();
      rows.applyToSelected([&](vector_size_t row) {
        if ((baseInMap == nullptr ||
             bits::isBitSet(baseInMap, decodedMap.index(row))) &&
            (rawRemoved == nullptr || !bits::isBitSet(rawRemoved, row))) {
          bits::setBit(rawInMap, row);
        }
      });
      inMaps.push_back(std::move(inMap));
    }
    // Top level nulls are left to the engine: map_except is a default-null
    // function, so no row in 'rows' has a null map.
  }

  auto localResult = std::make_shared<FlatMapVector>(
      context.pool(),
      outputType,
      std::move(nulls),
      numRows,
      flatMap.distinctKeys(),
      std::move(mapValues),
      std::move(inMaps));
  localResult->shareKeyIndex(flatMap);

  context.moveOrCopyResult(localResult, rows, result);
}

// Removes excluded keys from FlatMapVector inputs by clearing in-map bits and
// delegates all other encodings to the existing simple-function adapter.
class MapExceptVectorFunction : public exec::VectorFunction {
 public:
  explicit MapExceptVectorFunction(
      std::shared_ptr<exec::VectorFunction> simpleFunction)
      : simpleFunction_{std::move(simpleFunction)} {}

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& outputType,
      exec::EvalCtx& context,
      VectorPtr& result) const override {
    VELOX_CHECK_EQ(args.size(), 2, "map_except expects a map and an array.");

    // Probing with wrappedVector() rather than a DecodedVector leaves the
    // MapVector path with the single decode the simple function already does.
    if (args[0]->wrappedVector()->encoding() !=
        VectorEncoding::Simple::FLAT_MAP) {
      simpleFunction_->apply(rows, args, outputType, context, result);
      return;
    }

    exec::LocalDecodedVector mapDecoder(context, *args[0], rows);
    const auto& decodedMap = *mapDecoder.get();
    const auto& flatMap = *decodedMap.base()->asUnchecked<FlatMapVector>();

    // A flat map without keys has nothing to remove, and its key index cannot
    // be built.
    if (flatMap.numDistinctKeys() == 0) {
      context.moveOrCopyResult(args[0], rows, result);
      return;
    }

    // The simple functions match NaN keys to each other, which the flat map
    // key index does not. Flat maps rarely have floating point keys, so these
    // are converted to a MapVector instead.
    const auto keyKind = flatMap.keyType()->kind();
    if (keyKind == TypeKind::REAL || keyKind == TypeKind::DOUBLE) {
      VectorPtr map = flatMap.toMapVector();
      if (!decodedMap.isIdentityMapping()) {
        map = BaseVector::wrapInDictionary(
            BufferPtr(nullptr),
            copyIndices(decodedMap, rows, context.pool()),
            decodedMap.size(),
            std::move(map));
      }
      std::vector<VectorPtr> mapArgs{std::move(map), args[1]};
      simpleFunction_->apply(rows, mapArgs, outputType, context, result);
      return;
    }

    const bool sameRows = sharesRowSpace(decodedMap, rows);
    if (args[1]->isConstantEncoding() && sameRows) {
      applyFlatMap(
          args[0], args[1], flatMap, rows, outputType, context, result);
    } else {
      exec::LocalDecodedVector keysDecoder(context, *args[1], rows);
      applyFlatMap(
          args[0],
          decodedMap,
          sameRows,
          *keysDecoder.get(),
          flatMap,
          rows,
          outputType,
          context,
          result);
    }
  }

 private:
  const std::shared_ptr<exec::VectorFunction> simpleFunction_;
};

std::shared_ptr<exec::VectorFunction> makeMapExcept(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  std::vector<TypePtr> inputTypes;
  std::vector<VectorPtr> constantInputs;
  inputTypes.reserve(inputArgs.size());
  constantInputs.reserve(inputArgs.size());
  for (const auto& arg : inputArgs) {
    inputTypes.push_back(arg.type);
    constantInputs.push_back(arg.constantValue);
  }

  auto simpleFunction =
      exec::simpleFunctions().resolveFunction(name, inputTypes);
  VELOX_CHECK(
      simpleFunction.has_value(),
      "Scalar function not registered for the given input types: {}",
      name);

  return std::make_shared<MapExceptVectorFunction>(
      simpleFunction->createFunction()->createVectorFunction(
          inputTypes, constantInputs, config));
}

template <typename T>
void registerMapExceptPrimitive(
    const std::string& name,
    std::string_view defaultOwner) {
  registerFunction<
      ParameterBinder<MapExceptPrimitiveFunction, T>,
      Map<T, Generic<T1>>,
      Map<T, Generic<T1>>,
      Array<T>>({name}, true, defaultOwner);
}

} // namespace

void registerMapExcept(const std::string& name, std::string_view defaultOwner) {
  registerMapExceptPrimitive<bool>(name, defaultOwner);
  registerMapExceptPrimitive<int8_t>(name, defaultOwner);
  registerMapExceptPrimitive<int16_t>(name, defaultOwner);
  registerMapExceptPrimitive<int32_t>(name, defaultOwner);
  registerMapExceptPrimitive<int64_t>(name, defaultOwner);
  registerMapExceptPrimitive<float>(name, defaultOwner);
  registerMapExceptPrimitive<double>(name, defaultOwner);
  registerMapExceptPrimitive<Timestamp>(name, defaultOwner);
  registerMapExceptPrimitive<Date>(name, defaultOwner);

  registerFunction<
      MapExceptVarcharFunction,
      Map<Varchar, Generic<T1>>,
      Map<Varchar, Generic<T1>>,
      Array<Varchar>>({name}, {}, true, defaultOwner);

  // The simple functions only cover these key types, so the vector function
  // lists them rather than a type variable that would also match complex keys.
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  for (const auto& keyType :
       {"boolean",
        "tinyint",
        "smallint",
        "integer",
        "bigint",
        "real",
        "double",
        "timestamp",
        "date",
        "varchar"}) {
    // map(K,V), array(K) -> map(K,V)
    signatures.push_back(
        exec::FunctionSignatureBuilder()
            .typeVariable("V")
            .returnType(fmt::format("map({},V)", keyType))
            .argumentType(fmt::format("map({},V)", keyType))
            .argumentType(fmt::format("array({})", keyType))
            .build());
  }

  // The expression compiler prefers this vector function; its factory creates
  // the existing simple-function adapter as the non-flat-map fallback.
  exec::registerStatefulVectorFunction(
      name,
      std::move(signatures),
      [name](
          const std::string& /*name*/,
          const std::vector<exec::VectorFunctionArg>& inputArgs,
          const core::QueryConfig& config) {
        return makeMapExcept(name, inputArgs, config);
      },
      {},
      true,
      defaultOwner);
}

} // namespace facebook::velox::functions
