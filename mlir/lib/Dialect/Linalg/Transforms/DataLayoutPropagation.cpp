//===- DataLayoutPropagation.cpp -----------------------------------------===///
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_LINALGDATALAYOUTPROPAGATION
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "linalg-data-layout-propagation"

namespace {

/// Returns a tuple for packed operand and indexing_map with the assumptions:
///   1) The generic op is the producer of the pack op.
///   2) The generic op has only one result.
///   3) The indexing map of the output operand is identity.
/// If the operand is a scalar or packing dimensions are all irrelevant to the
/// operand, the opreand and the updated indexing map will be returned.
/// Otherwise, it returns the packed operand and the updated indexing map. E.g.,
///
///   #map0 = affine_map<(d0, d1) -> (d0, d1)>
///   #map1 = affine_map<(d0, d1) -> (d0)>
///   #map2 = affine_map<(d0, d1) -> (d1)>
///   %0 = linalg.generic {indexing_maps = [#map1, #map2, #map0],
///                        iterator_types = ["parallel", "parallel"]}
///      ins(%arg0, %arg1 : tensor<?xf32>, tensor<?xf32>)
///      outs(%init : tensor<?x?xf32>) {
///    ^bb0(%arg3: f32, %arg4: f32, %arg5: f32):
///      %4 = arith.addf %arg3, %arg4 : f32
///      linalg.yield %4 : f32
///  } -> tensor<?x?xf32>
///  %1 = tensor.pack %0
///    inner_dims_pos = [0, 1]
///    inner_tiles = [8, 2]
///    into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///
///  Taking the first input operand as an example, the inner tile size of d1 is
///  8. Thus, the below operation and `affine_map<(d0, d1, d2, d3)> ->
///  affine_map<(d1, d3)>` will be returned.
///
///  %pack = tensor.pack %arg0
///    inner_dims_pos = [0]
///    inner_tiles = [8]
///    into %init : tensor<?xf32> -> tensor<?x8xf32>
static std::tuple<Value, AffineMap>
getOrCreatePackedViewOfOperand(OpBuilder &b, Location loc,
                               tensor::PackOp packOp, GenericOp genericOp,
                               OpOperand *opOperand) {
  int numOrigLoops = genericOp.getNumLoops();
  int64_t numInnerLoops = packOp.getInnerDimsPos().size();
  int64_t numLoops = numOrigLoops + numInnerLoops;
  AffineMap origIndexingMap = genericOp.getMatchingIndexingMap(opOperand);
  SmallVector<AffineExpr> exprs(origIndexingMap.getResults());

  if (genericOp.isScalar(opOperand))
    return std::make_tuple(
        opOperand->get(),
        AffineMap::get(numLoops, 0, exprs, packOp.getContext()));

  llvm::SetVector<int64_t> innerDimsPosSet(packOp.getInnerDimsPos().begin(),
                                           packOp.getInnerDimsPos().end());
  // Mapping from AffinDimExpr of indexing maps to the operand shape dimension.
  DenseMap<int64_t, int64_t> iterMapToDim;
  for (auto [index, expr] : llvm::enumerate(origIndexingMap.getResults())) {
    int64_t dimPos = expr.cast<AffineDimExpr>().getPosition();
    if (!innerDimsPosSet.contains(dimPos))
      continue;
    iterMapToDim[dimPos] = index;
  }

  // Construct the information of packing data dimensions and new indexing maps
  // for the operand.
  SmallVector<int64_t> innerDimsPos;
  SmallVector<OpFoldResult> innerTileSizes;
  for (auto [index, value] : llvm::enumerate(
           llvm::zip(packOp.getInnerDimsPos(), packOp.getMixedTiles()))) {
    int64_t dimPos = std::get<0>(value);
    if (!iterMapToDim.count(dimPos))
      continue;
    innerDimsPos.push_back(iterMapToDim[dimPos]);
    innerTileSizes.push_back(std::get<1>(value));
    exprs.push_back(b.getAffineDimExpr(numOrigLoops + index));
  }
  auto indexingMap = AffineMap::get(numLoops, 0, exprs, packOp.getContext());

  SmallVector<int64_t> outerDimsPerm;
  for (auto outDim : packOp.getOuterDimsPerm()) {
    if (!iterMapToDim.count(outDim))
      continue;
    outerDimsPerm.push_back(iterMapToDim[outDim]);
  }

  // The operand does not have dimensions that relates to pack op.
  if (innerDimsPos.empty() && outerDimsPerm.empty())
    return std::make_tuple(opOperand->get(), indexingMap);

  auto empty = tensor::PackOp::createDestinationTensor(
      b, loc, opOperand->get(), innerTileSizes, innerDimsPos, outerDimsPerm);
  auto packedOperand = b.create<tensor::PackOp>(
      loc, opOperand->get(), empty, innerDimsPos, innerTileSizes,
      packOp.getPaddingValue(), outerDimsPerm);
  return std::make_tuple(packedOperand, indexingMap);
}

/// Bubbles up tensor.pack op through elementwise generic op. This
/// swap pack(generic) to generic(pack). The new generic op works on packed
/// domain; pack ops are created for input and output operands. E.g.,
///
///     #map0 = affine_map<(d0, d1) -> (d0, d1)>
///     %0 = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///     %1 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
///     %2 = tensor.empty(%0, %1) : tensor<?x?xf32>
///     %3 = linalg.generic {indexing_maps = [#map0, #map0],
///                          iterator_types = ["parallel", "parallel"]}
///         ins(%arg0 : tensor<?x?xf32>)
///         outs(%2 : tensor<?x?xf32>) {
///       ^bb0(%arg3: f32, %arg4: f32):
///         %4 = arith.addf %arg3, %arg3 : f32
///         linalg.yield %4 : f32
///     } -> tensor<?x?xf32>
///     %4 = tensor.pack %3
///       inner_dims_pos = [0, 1]
///       inner_tiles = [8, 2]
///       into %dest : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///
/// will be converted to
///
///     #map = affine_map<()[s0] -> (s0 ceildiv 8)>
///     #map1 = affine_map<()[s0] -> (s0 ceildiv 2)>
///     #map2 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
///     %dim = tensor.dim %arg0, %c0 : tensor<?x?xf32>
///     %dim_0 = tensor.dim %arg0, %c1 : tensor<?x?xf32>
///     %0 = affine.apply #map()[%dim]
///     %1 = affine.apply #map1()[%dim_0]
///     %2 = tensor.empty(%0, %1) : tensor<?x?x8x2xf32>
///     %pack = tensor.pack %arg0
///       inner_dims_pos = [0, 1]
///       inner_tiles = [8, 2]
///       into %2 : tensor<?x?xf32> -> tensor<?x?x8x2xf32>
///     %3 = linalg.generic {indexing_maps = [#map2, #map2],
///       iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
///       ins(%pack : tensor<?x?x8x2xf32>)
///       outs(%arg1 : tensor<?x?x8x2xf32>) {
///     ^bb0(%in: f32, %out: f32):
///       %4 = arith.addf %in, %in : f32
///       linalg.yield %4 : f32
///     } -> tensor<?x?x8x2xf32>
static FailureOr<GenericOp>
bubbleUpPackOpThroughElemGenericOp(RewriterBase &rewriter,
                                   tensor::PackOp packOp) {
  auto genericOp = packOp.getSource().getDefiningOp<GenericOp>();
  if (!genericOp)
    return failure();

  if (!isElementwise(genericOp))
    return failure();

  // TODO: Relax the restriction. We are able to bubble up the pack op through
  // multi-result generic op. It just needs more work.
  if (genericOp.getNumResults() != 1)
    return failure();

  // TODO: Add an option for allowing padding values. It could introduce
  // undefined behavior if we unconditionally propagate pack op through all
  // the ops. E.g., if the padding value is zero and there are division ops in
  // a generic op. Some values of padding area could be NaN (0/0).
  if (packOp.getPaddingValue())
    return failure();

  OpOperand *opOperand = genericOp.getDpsInitOperand(0);
  // TODO: Add support for all permutation indexing maps.
  if (!genericOp.getMatchingIndexingMap(opOperand).isIdentity())
    return rewriter.notifyMatchFailure(
        packOp, "the result of generic op does not have identity indexing_map");

  Location loc = packOp.getLoc();
  SmallVector<Value> inputOperands;
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    auto [packedOperand, packedIndexingMap] = getOrCreatePackedViewOfOperand(
        rewriter, loc, packOp, genericOp, inputOperand);
    inputOperands.push_back(packedOperand);
    indexingMaps.push_back(packedIndexingMap);
  }

  int64_t numLoops = genericOp.getNumLoops();
  int64_t numInnerLoops = packOp.getInnerDimsPos().size();
  int64_t newNumLoops = numLoops + numInnerLoops;
  SmallVector<utils::IteratorType> iterTypes =
      genericOp.getIteratorTypesArray();
  iterTypes.append(numInnerLoops, utils::IteratorType::parallel);

  SmallVector<AffineExpr> outExprs(
      genericOp.getMatchingIndexingMap(opOperand).getResults());
  for (int i = 0; i < numInnerLoops; ++i)
    outExprs.push_back(rewriter.getAffineDimExpr(numLoops + i));
  indexingMaps.push_back(
      AffineMap::get(newNumLoops, 0, outExprs, rewriter.getContext()));

  auto newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, packOp.getDestType(), inputOperands, packOp.getDest(), indexingMaps,
      iterTypes, /*bodyBuild=*/nullptr,
      linalg::getPrunedAttributeList(genericOp));
  rewriter.inlineRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(),
                              newGenericOp.getRegion().begin());
  return newGenericOp;
}

// Wrapper pattern that applies bubbleUpPackOpThroughElemGenericOp method.
class BubbleUpPackOpThroughElemGenericOpPattern
    : public OpRewritePattern<tensor::PackOp> {
public:
  using OpRewritePattern<tensor::PackOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(tensor::PackOp packOp,
                                PatternRewriter &rewriter) const override {
    auto genericOp = bubbleUpPackOpThroughElemGenericOp(rewriter, packOp);
    if (failed(genericOp))
      return failure();
    rewriter.replaceOp(packOp, genericOp.value().getResults());
    return success();
  }
};

struct PackInfo {
  int64_t getNumTiledLoops() const { return packedDims.size(); };

  llvm::DenseMap<int64_t, OpFoldResult> packedDims;
  llvm::SetVector<int64_t> loopPerm;
};

template <typename T, unsigned N>
void applyPermutationToVectorImpl(SmallVector<T, N> &inVec,
                              ArrayRef<int64_t> permutation) {
  SmallVector<T, N> auxVec = inVec;
  for (const auto &en : enumerate(permutation))
    auxVec[en.index()] = inVec[en.value()];
  inVec = auxVec;
}

static std::tuple<Value, AffineMap>
getOrCreatePackedViewOfOperandTest(OpBuilder &builder, Location loc,
                                   PackInfo &packInfo, GenericOp genericOp,
                                   OpOperand *opOperand) {
  int numOrigLoops = genericOp.getNumLoops();
  int64_t numLoops = numOrigLoops + packInfo.getNumTiledLoops();

  AffineMap origIndexingMap = genericOp.getMatchingIndexingMap(opOperand);
  SmallVector<AffineExpr> exprs(origIndexingMap.getResults());

  if (genericOp.isScalar(opOperand))
    return std::make_tuple(
        opOperand->get(),
        AffineMap::get(numLoops, 0, exprs, genericOp.getContext()));

  // Handle `outer_dims_perm`. See example:
  // current map      : (d0, d1, d2, d3) -> (d2, d3)
  // dimAndTileMapping: dim | tile
  //                    3   | 32
  // tileLoopPerm     : [0, 3, 1, 2, 4]
  // First map d2, d3 with their position in the array as:
  // currentPositionTileLoops: dim | pos
  //                           d2  | 0
  //                           d3  | 1
  // then scan `tileLoopPerm` in order and get the `outer_dims_perm`
  // to be used, here it would be [1, 0].
  SmallVector<int64_t> outerDimsPerm;
  DenseMap<int64_t, int64_t> iterMapToDim;
  for (auto [index, expr] : llvm::enumerate(origIndexingMap.getResults())) {
    iterMapToDim[expr.cast<AffineDimExpr>().getPosition()] = index;
  }
  for (int64_t loopIdx : packInfo.loopPerm) {
    if (iterMapToDim.count(loopIdx))
      outerDimsPerm.push_back(iterMapToDim[loopIdx]);
  }

  // Handle `inner_dims_pos`
  SmallVector<int64_t> innerDimsPos;
  SmallVector<OpFoldResult> innerTileSizes;
  int64_t currentDimIdx = 0;
  for (auto [index, expr] : llvm::enumerate(origIndexingMap.getResults())) {
    int64_t dimPos = expr.cast<AffineDimExpr>().getPosition();
    if (!packInfo.packedDims.count(dimPos))
      continue;
    innerTileSizes.push_back(packInfo.packedDims[dimPos]);
    innerDimsPos.push_back(index);
    exprs.push_back(builder.getAffineDimExpr(numOrigLoops + currentDimIdx++));
  }
 
  applyPermutationToVectorImpl(exprs, outerDimsPerm);
  for (AffineExpr e : exprs)
   e.dump(); 
  
  auto indexingMap = AffineMap::get(numLoops, 0, exprs, genericOp.getContext());
  AffineMap permutationMap = AffineMap::getPermutationMap(
      SmallVector<unsigned>(packInfo.loopPerm.getArrayRef().begin(),
                            packInfo.loopPerm.getArrayRef().end()),
      genericOp.getContext());
  AffineMap invPerm = inversePermutation(permutationMap);

  llvm::errs() << "=================================\n";
  for (int64_t outer : outerDimsPerm)
    llvm::errs() << outer << " ";
  llvm::errs() << "\n";
  llvm::errs() << "map: " << indexingMap << "\n";
  llvm::errs() << "perm: " << permutationMap << "\n";
  llvm::errs() << "invPerm: " << invPerm << "\n";
  llvm::errs() << "composed: " << indexingMap.compose(invPerm) << "\n";
  llvm::errs() << "=================================\n";
  indexingMap = indexingMap.compose(invPerm);

  // The operand does not have dimensions that relates to pack op.
  if (innerDimsPos.empty() && outerDimsPerm.empty())
    return std::make_tuple(opOperand->get(), indexingMap);

  // Simply forward the argument if it comes from an unpack.
  tensor::UnPackOp sourceDefOperand =
      opOperand->get().getDefiningOp<tensor::UnPackOp>();
  if (sourceDefOperand) {
    return std::make_tuple(sourceDefOperand.getSource(), indexingMap);
  }
  // Otherwise pack it.
  auto empty = tensor::PackOp::createDestinationTensor(
      builder, loc, opOperand->get(), innerTileSizes, innerDimsPos,
      outerDimsPerm);
  auto packedOperand =
      builder.create<tensor::PackOp>(loc, opOperand->get(), empty, innerDimsPos,
                                     innerTileSizes, llvm::None, outerDimsPerm);
  return std::make_tuple(packedOperand, indexingMap);
}

// Extract packing information from the generic. Fail if we find conflicting
// information.
static FailureOr<PackInfo> getPackingInfo(linalg::GenericOp genericOp) {
  PackInfo packInfo;
  for (OpOperand &operand : genericOp->getOpOperands()) {
    tensor::UnPackOp unpackOp = operand.get().getDefiningOp<tensor::UnPackOp>();
    if (!unpackOp)
      continue;

    // Find which dimension in the domain are tiled by looking at the operand
    // codomain.
    AffineMap indexingMap = genericOp.getMatchingIndexingMap(&operand);
    SmallVector<AffineExpr> exprs(indexingMap.getResults());
    llvm::DenseSet<int64_t> innerDimsPosSet(unpackOp.getInnerDimsPos().begin(),
                                            unpackOp.getInnerDimsPos().end());
    size_t idxInTiles = 0;
    for (auto [index, expr] : llvm::enumerate(indexingMap.getResults())) {
      int64_t dimPos = expr.cast<AffineDimExpr>().getPosition();
      // If index is in `innerDimsPosSet` the current tensor dimension is tiled.
      // Get the position in the domain and bind it to the tile size.
      if (!innerDimsPosSet.contains(index))
        continue;
      packInfo.packedDims[dimPos] = unpackOp.getMixedTiles()[idxInTiles++];
    }

    // Get an `outer_dims_perm` permutation in the domain
    ArrayRef<int64_t> outerDimsPerm = unpackOp.getOuterDimsPerm();
    for (int64_t dim : outerDimsPerm)
      packInfo.loopPerm.insert(indexingMap.getDimPosition(dim));
    for (int64_t dimPos : unpackOp.getInnerDimsPos())
      packInfo.loopPerm.insert(dimPos + unpackOp.getInnerDimsPos().size());
  }

  // No work to do.
  if (packInfo.packedDims.size() == 0)
    return failure();
  return packInfo;
}

// %0 = unpack t1 in t2
// %1 = linalg.elemwse ins(   ) outs(   ) [single output]
//
// Case1. All operands come from `unpack` operations -> I have all the
// information Case2. Some operands from `unpack` operations -> Partial
// information
//
FailureOr<GenericOp>
pushDownUnPackOpThroughElemGenericOp(RewriterBase &rewriter,
                                     GenericOp genericOp) {
  if (!isElementwise(genericOp))
    return failure();

  if (genericOp.getNumResults() != 1)
    return failure();

  OpOperand *opOperand = genericOp.getDpsInitOperand(0);
  if (!genericOp.getMatchingIndexingMap(opOperand).isIdentity())
    return failure();

  FailureOr<PackInfo> packInfo = getPackingInfo(genericOp);
  if (failed(packInfo))
    return failure();

  llvm::errs() << "=================================\n";
  llvm::errs() << genericOp << "\n";
  llvm::errs() << "loop perm: ";
  for (unsigned u : packInfo.value().loopPerm)
    llvm::errs() << u << " ";
  llvm::errs() << "\n";
  llvm::errs() << "=================================\n";

  Location loc = genericOp.getLoc();
  SmallVector<Value> packedInputOperands;
  SmallVector<AffineMap> packedIndexingMaps;
  for (OpOperand *operand : genericOp.getDpsInputOperands()) {
    auto [packedOperand, packedIndexingMap] =
        getOrCreatePackedViewOfOperandTest(rewriter, loc, packInfo.value(),
                                           genericOp, operand);
    packedInputOperands.push_back(packedOperand);
    packedIndexingMaps.push_back(packedIndexingMap);
  }

  Value packedOutputOperand;
  auto [packedOperand, packedIndexingMap] = getOrCreatePackedViewOfOperandTest(
      rewriter, loc, packInfo.value(), genericOp,
      genericOp.getDpsInitOperand(0));
  packedOutputOperand = packedOperand;
  packedIndexingMaps.push_back(packedIndexingMap);

  llvm::errs() << "=================================\n";
  llvm::errs() << packedOutputOperand << "\n";
  for (Value packedInput : packedInputOperands)
    llvm::errs() << packedInput << "\n";
  llvm::errs() << "=================================\n";

  int64_t numInnerLoops = packInfo.value().getNumTiledLoops();
  SmallVector<utils::IteratorType> iterTypes =
      genericOp.getIteratorTypesArray();
  iterTypes.append(numInnerLoops, utils::IteratorType::parallel);

  auto newGenericOp = rewriter.create<GenericOp>(
      loc, packedOutputOperand.getType(), packedInputOperands,
      packedOutputOperand, packedIndexingMaps, iterTypes, /*bodyBuild=*/nullptr,
      linalg::getPrunedAttributeList(genericOp));
  rewriter.inlineRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(),
                              newGenericOp.getRegion().begin());

  // re-create the unpack after the new generic operation.
  Value newGenericOpResult = newGenericOp.getDpsInitOperand(0)->get();
  tensor::PackOp packOp = newGenericOpResult.getDefiningOp<tensor::PackOp>();
  // The original output was not unpacked, thus a pack is inserted by
  // `getOrCreatePackedViewOfOperandTest`.
  if (packOp) {
    tensor::UnPackOp newUnpackOp = rewriter.create<tensor::UnPackOp>(
        loc, packOp.getSource().getType(),
        newGenericOp.getTiedOpResult(newGenericOp.getDpsInitOperand(0)),
        packOp.getSource(), packOp.getOuterDimsPerm(), packOp.getInnerDimsPos(),
        packOp.getInnerTiles(), packOp.getStaticTiles());
    newGenericOpResult = newUnpackOp.getResult();
  }
  // The original output was unpacked.
  else {
    tensor::UnPackOp unpackOp =
        genericOp.getDpsInitOperand(0)->get().getDefiningOp<tensor::UnPackOp>();
    assert(unpackOp && "must be valid");
    tensor::UnPackOp newUnpackOp = rewriter.create<tensor::UnPackOp>(
        loc, unpackOp.getDest().getType(),
        newGenericOp.getTiedOpResult(newGenericOp.getDpsInitOperand(0)),
        unpackOp.getDest(), unpackOp.getOuterDimsPerm(),
        unpackOp.getInnerDimsPos(), unpackOp.getInnerTiles(),
        unpackOp.getStaticTiles());
    newGenericOpResult = newUnpackOp.getResult();
  }
  rewriter.replaceOp(genericOp, newGenericOpResult);
  return genericOp;
}

// Wrapper pattern that applies pushDownUnPackOpThroughElemGenericOp method.
struct PushDownUnPackOpThroughElemGenericOpPattern
    : public OpRewritePattern<GenericOp> {
  using OpRewritePattern<GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(GenericOp genericOp,
                                PatternRewriter &rewriter) const override {
    auto newGenericOp =
        pushDownUnPackOpThroughElemGenericOp(rewriter, genericOp);
    if (failed(newGenericOp))
      return failure();
    // rewriter.replaceOp(genericOp, newGenericOp.value().getResults());
    return success();
  }
};

} // namespace

void mlir::linalg::populateDataLayoutPropagationPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<BubbleUpPackOpThroughElemGenericOpPattern,
                  PushDownUnPackOpThroughElemGenericOpPattern>(
      patterns.getContext());
}
