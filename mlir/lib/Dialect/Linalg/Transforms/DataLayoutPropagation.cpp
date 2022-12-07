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

struct PackInfo {
  int64_t getNumTiledLoopsOperand(AffineMap map) const {
    int64_t cnt = 0;
    for (AffineExpr expr : map.getResults()) {
      int64_t pos = expr.cast<AffineDimExpr>().getPosition();
      if (!packedDims.count(pos))
        continue;
      cnt++;
    }
    return cnt;
  };
  int64_t getNumTiledLoops() const { return packedDims.size(); };
  // domain -> tile for tiled loops.
  llvm::DenseMap<int64_t, OpFoldResult> packedDims;
  // entire domain loop perm only for outer loops.
  // if we have any permutation on the point loop it is propagated in the pack.
  llvm::SmallVector<int64_t> loopPerm;

  llvm::SmallVector<int64_t> innerDimsPos;
};

template <typename OpTy>
PackInfo getPackingInfo(AffineMap indexingMap, OpTy packOp) {
  PackInfo packInfo;
  SmallVector<AffineExpr> exprs(indexingMap.getResults());
  llvm::DenseSet<int64_t> innerDimsPosSet(packOp.getInnerDimsPos().begin(),
                                          packOp.getInnerDimsPos().end());
  size_t idxInTiles = 0;
  size_t idxInDimPos = 0;
  for (auto [index, expr] : llvm::enumerate(indexingMap.getResults())) {
    int64_t dimPos = expr.template cast<AffineDimExpr>().getPosition();
    // If index is in `innerDimsPosSet` the current tensor dimension is tiled.
    // Get the position in the domain and bind it to the tile size.
    if (!innerDimsPosSet.contains(index))
      continue;
    packInfo.packedDims[dimPos] = packOp.getMixedTiles()[idxInTiles++];
  }
  // build loop perm, assume no permutation.
  size_t idx = 0;
  while (idx < packInfo.getNumTiledLoops() + indexingMap.getNumDims())
    packInfo.loopPerm.push_back(idx++);
  assert(packInfo.loopPerm.size() ==
         packInfo.getNumTiledLoops() + indexingMap.getNumDims());

  // if we have permutation by outer dims add them.
  idx = 0;
  ArrayRef<int64_t> outerDimsPerm = packOp.getOuterDimsPerm();
  for (int64_t dim : outerDimsPerm)
    packInfo.loopPerm[idx++] = indexingMap.getDimPosition(dim);

  assert(packInfo.loopPerm.size() ==
         packInfo.getNumTiledLoops() + indexingMap.getNumDims());

  packInfo.innerDimsPos = llvm::to_vector(packOp.getInnerDimsPos());
  return packInfo;
}

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

SmallVector<int64_t> getInnerDimsPosForOperand(AffineMap origIndexingMap,
                                               PackInfo &packInfo) {
  llvm::DenseMap<int64_t, int64_t> domainToDimOperandPos;
  for (auto [index, expr] : llvm::enumerate(origIndexingMap.getResults())) {
    int64_t pos = expr.cast<AffineDimExpr>().getPosition();
    domainToDimOperandPos[pos] = index;
  }
  SmallVector<int64_t> innerDimsPos;
  for (int64_t dimPos : packInfo.innerDimsPos) {
    if (!domainToDimOperandPos.count(dimPos)) continue;
    int64_t index = domainToDimOperandPos[dimPos];
    innerDimsPos.push_back(index);
  }
  return innerDimsPos;
}

static std::tuple<Value, AffineMap>
getOrCreatePackedViewOfOperand(OpBuilder &b, Location loc, PackInfo &packInfo,
                               GenericOp genericOp, OpOperand *opOperand) {
  int numOrigLoops = genericOp.getNumLoops();
  AffineMap origIndexingMap = genericOp.getMatchingIndexingMap(opOperand);
  int origRes = origIndexingMap.getNumResults();
  SmallVector<AffineExpr> exprs(origIndexingMap.getResults());
  MLIRContext *ctx = genericOp.getContext();
  int64_t numLoops = numOrigLoops + packInfo.getNumTiledLoops();

  if (genericOp.isScalar(opOperand))
    return std::make_tuple(opOperand->get(),
                           AffineMap::get(numLoops, 0, exprs, ctx));

  llvm::errs() << "=================================\n";
  llvm::errs() << "--> operand map: " << origIndexingMap << "\n";
  llvm::errs() << "loop permutation: \n";
  for (int64_t loopIdx : packInfo.loopPerm)
    llvm::errs() << loopIdx << " ";
  llvm::errs() << "\n";

  // We fold the transpose into the generic no outer dims perm for the pack.
  // This is bad for later fusion.
  SmallVector<int64_t> outerDimsPerm = {};

  SmallVector<OpFoldResult> innerTileSizes;
  for (auto [index, expr] : llvm::enumerate(origIndexingMap.getResults())) {
    int64_t dimPos = expr.cast<AffineDimExpr>().getPosition();
    if (!packInfo.packedDims.count(dimPos))
      continue;
    innerTileSizes.push_back(packInfo.packedDims[dimPos]);
    // point loops are simply appended. The permutation is pushed out into the
    // pack.
    exprs.push_back(b.getAffineDimExpr(dimPos + packInfo.getNumTiledLoops()));
  }

  SmallVector<int64_t> innerDimsPos =
      getInnerDimsPosForOperand(origIndexingMap, packInfo);

  llvm::errs() << "outer perm: \n";
  for (int64_t outer : outerDimsPerm)
    llvm::errs() << outer << " ";
  llvm::errs() << "\n";

  auto indexingMap = AffineMap::get(numLoops, 0, exprs, genericOp.getContext());
  llvm::errs() << "new indexing map:" << indexingMap << "\n";
  AffineMap permutationMap = AffineMap::getPermutationMap(
      SmallVector<unsigned>(packInfo.loopPerm.begin(), packInfo.loopPerm.end()),
      ctx);
  llvm::errs() << "perm map: " << permutationMap << "\n";
  AffineMap invPerm = inversePermutation(permutationMap);
  llvm::errs() << "inverse map: " << invPerm << "\n";
  indexingMap = indexingMap.compose(invPerm);
  llvm::errs() << "new map: " << indexingMap << "\n";
  llvm::errs() << "inner pos: \n";
  for (int64_t inner : innerDimsPos)
    llvm::errs() << inner << " ";
  llvm::errs() << "\n";
  llvm::errs() << "=================================\n";

  // The operand does not have dimensions that relates to pack op.
  if (innerDimsPos.empty() && outerDimsPerm.empty())
    return std::make_tuple(opOperand->get(), indexingMap);

  auto empty = tensor::PackOp::createDestinationTensor(
      b, loc, opOperand->get(), innerTileSizes, innerDimsPos, outerDimsPerm);
  auto packedOperand =
      b.create<tensor::PackOp>(loc, opOperand->get(), empty, innerDimsPos,
                               innerTileSizes, llvm::None, outerDimsPerm);
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

  auto packInfo =
      getPackingInfo(genericOp.getMatchingIndexingMap(opOperand), packOp);

  Location loc = packOp.getLoc();
  SmallVector<Value> inputOperands;
  SmallVector<AffineMap> indexingMaps;
  for (OpOperand *inputOperand : genericOp.getDpsInputOperands()) {
    auto [packedOperand, packedIndexingMap] = getOrCreatePackedViewOfOperand(
        rewriter, loc, packInfo, genericOp, inputOperand);
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
  AffineMap outMap =
      AffineMap::get(newNumLoops, 0, outExprs, rewriter.getContext());
  // llvm::errs() << "output map: " << outMap << "\n";
  indexingMaps.push_back(outMap);

  auto newGenericOp = rewriter.create<linalg::GenericOp>(
      loc, packOp.getDestType(), inputOperands, packOp.getDest(), indexingMaps,
      iterTypes, /*bodyBuild=*/nullptr,
      linalg::getPrunedAttributeList(genericOp));
  rewriter.cloneRegionBefore(genericOp.getRegion(), newGenericOp.getRegion(),
                             newGenericOp.getRegion().begin());
  return newGenericOp;
}

// Wrapper pattern that applies bubbleUpPackOpThroughElemGenericOp method.
struct BubbleUpPackOpThroughElemGenericOpPattern
    : public OpRewritePattern<tensor::PackOp> {
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
} // namespace

void mlir::linalg::populateDataLayoutPropagationPatterns(
    RewritePatternSet &patterns) {
  patterns.insert<BubbleUpPackOpThroughElemGenericOpPattern>(
      patterns.getContext());
}
