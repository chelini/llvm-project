//===- ChainOpt.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/Support/Debug.h"
#include <queue>

#define DEBUG_TYPE "matmul-chain"

using namespace mlir;

namespace {

struct LinalgChainOptPass : public LinalgChainOptBase<LinalgChainOptPass> {

  // Check 'out' to be the left operand for 'operation'.
  // 'out' is the value produced by another matmul operation.
  bool isLeftOperand(Value out, Operation *operation) {
    if (auto chainedMulOp = dyn_cast_or_null<linalg::MatmulOp>(operation)) {
      Value leftOperand = chainedMulOp.inputs()[0];
      if (leftOperand == out)
        return true;
      Value rightOperand = chainedMulOp.inputs()[1];
      if (rightOperand == out)
        return false;
    }
    return false;
  }

  // return the static shape for a ranked memref operand on success.
  LogicalResult getStaticShape(Type operand, SmallVectorImpl<int64_t> &shape) {
    if (MemRefType memref = operand.dyn_cast_or_null<MemRefType>()) {
      if (!memref.hasStaticShape()) {
        return failure();
      }
      assert(memref.getShape().size() == 2);
      shape.push_back(memref.getShape()[0]);
      shape.push_back(memref.getShape()[1]);
      return success();
    }
    return failure();
  }

  // collect the chain size and the operands. All the dimensions should
  // be statically known. We collect the operands to re-create the chain.
  LogicalResult getChainSizesAndOperands(ArrayRef<Operation *> chain,
                                         SmallVectorImpl<long> &chainSizes,
                                         SmallVectorImpl<Value> &operands) {
    assert(chain.size() > 1 && "must be a chain");
    Operation *head = chain[0];
    auto mulOp = dyn_cast_or_null<linalg::MatmulOp>(head);
    assert(mulOp && "must be a linalg::matmul");

    SmallVector<int64_t, 2> leftOperandShape;
    if (failed(getStaticShape(mulOp.inputs()[0].getType(), leftOperandShape)))
      return failure();
    SmallVector<int64_t, 2> rightOperandShape;
    if (failed(getStaticShape(mulOp.inputs()[1].getType(), rightOperandShape)))
      return failure();
    chainSizes.push_back(leftOperandShape[0]);
    chainSizes.push_back(leftOperandShape[1]);
    chainSizes.push_back(rightOperandShape[1]);
    operands.push_back(mulOp.inputs()[0]);
    operands.push_back(mulOp.inputs()[1]);

    // collect the remaining sizes considering
    // only the righ operand (second dimension).
    // 'isLeftOperand' match only this pattern.
    for (size_t i = 1; i < chain.size(); i++) {
      auto mulOp = dyn_cast_or_null<linalg::MatmulOp>(chain[i]);
      assert(mulOp && "must be a linalg::matmul");
      SmallVector<int64_t, 2> rightOperandShape;
      if (failed(
              getStaticShape(mulOp.inputs()[1].getType(), rightOperandShape)))
        return failure();
      operands.push_back(mulOp.inputs()[1]);
      chainSizes.push_back(rightOperandShape[1]);
    }

    return success();
  }

  // compute m and s table: See introduction to algorithm.
  void matrixChainOrder(ArrayRef<long> p, std::vector<std::vector<long>> &m,
                        std::vector<std::vector<long>> &s) {
    size_t n = p.size();
    for (size_t i = 0; i < n; i++)
      m[i][i] = 0;

    size_t j = 0;
    long q = 0;
    for (size_t l = 2; l < n; l++) {
      for (size_t i = 1; i < n - l + 1; i++) {
        j = i + l - 1;
        m[i][j] = std::numeric_limits<long>::max();
        for (size_t k = i; k <= j - 1; k++) {
          q = m[i][k] + m[k + 1][j] + p[i - 1] * p[k] * p[j];
          if (q < m[i][j]) {
            m[i][j] = q;
            s[i][j] = k;
          }
        }
      }
    }
  }

  // print optimal parentization.
  void printOptimalParensImpl(const std::vector<std::vector<long>> &s, size_t i,
                              size_t j) {
    if (i == j)
      llvm::dbgs() << " O_" << i << " ";
    else {
      llvm::dbgs() << "(";
      printOptimalParensImpl(s, i, s[i][j]);
      printOptimalParensImpl(s, s[i][j] + 1, j);
      llvm::dbgs() << ")";
    }
  }

  // print optimal parentization.
  void printOptimalParens(const std::vector<std::vector<long>> &s, size_t i,
                          size_t j) {
    llvm::dbgs() << "-------------------------------------\n";
    printOptimalParensImpl(s, i, j);
    llvm::dbgs() << "\n-------------------------------------\n";
  }

  Value rebuildChainImpl(ArrayRef<Value> operands, OpBuilder &builder,
                         Location loc, const std::vector<std::vector<long>> &s,
                         size_t i, size_t j) {
    if (i == j)
      return operands[i - 1];
    else {
      Value left = rebuildChainImpl(operands, builder, loc, s, i, s[i][j]);
      Value right = rebuildChainImpl(operands, builder, loc, s, s[i][j] + 1, j);

      MemRefType leftType = left.getType().dyn_cast_or_null<MemRefType>();
      assert(leftType && "expect to be a memref");
      MemRefType rightType = right.getType().dyn_cast_or_null<MemRefType>();
      assert(rightType && "expect to be a memref");

      SmallVector<int64_t, 2> shape = {leftType.getShape()[0],
                                       rightType.getShape()[1]};
      MemRefType resultType = MemRefType::get(shape, leftType.getElementType());

      // Materialize result.
      Value buffer = builder.create<memref::AllocOp>(loc, resultType);

      Attribute resultZeroAttr =
          builder.getZeroAttr(resultType.getElementType());
      Value zero = builder.create<arith::ConstantOp>(loc, resultZeroAttr);
      Value zeroIdx = builder.create<arith::ConstantIndexOp>(loc, 0);
      Value oneIdx = builder.create<arith::ConstantIndexOp>(loc, 1);

      SmallVector<Value> ubs;
      Value dim1 =
          builder.create<arith::ConstantIndexOp>(loc, resultType.getShape()[0]);
      Value dim2 =
          builder.create<arith::ConstantIndexOp>(loc, resultType.getShape()[1]);
      ubs = {dim1, dim2};
      SmallVector<Value> lbs = {zeroIdx, zeroIdx};
      SmallVector<Value> steps = {oneIdx, oneIdx};

      (void)scf::buildLoopNest(
          builder, loc, lbs, ubs, steps,
          [&](OpBuilder &b, Location loc, ValueRange localIvs) {
            b.create<memref::StoreOp>(loc, zero, buffer, localIvs);
          });

      Operation *op = builder.create<linalg::MatmulOp>(
          loc, ValueRange{left, right}, buffer);
      // skip walking the just created operation.
      visited.insert(op);
      return buffer;
    }
  }

  // we are ready to emit IR.
  void rebuildChain(ArrayRef<Operation *> chain, ArrayRef<Value> operands,
                    const std::vector<std::vector<long>> &s) {
    OpBuilder builder(chain[chain.size() - 1]->getContext());
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(chain[chain.size() - 1]);
    Location loc = chain[chain.size() - 1]->getLoc();
    Value val =
        rebuildChainImpl(operands, builder, loc, s, 1, chain.size() + 1);

    Operation *tail = chain[chain.size() - 1];
    auto lastOrigMatmul = dyn_cast_or_null<linalg::MatmulOp>(tail);
    assert(lastOrigMatmul && "must be a linalg::matmul");
    Value origOutputBuffer = lastOrigMatmul.outputs()[0];
    builder.create<memref::CopyOp>(loc, val, origOutputBuffer);
  }

  // optimize chain.
  void optimizeChainImpl(ArrayRef<Operation *> chain) {
    assert(chain.size() > 1 && "must be a chain!");
    SmallVector<long, 10> chainSizes;
    SmallVector<Value, 10> operands;
    if (failed(getChainSizesAndOperands(chain, chainSizes, operands)))
      return;
    const size_t n = chainSizes.size();
    std::vector<std::vector<long>> m(
        n, std::vector<long>(n, std::numeric_limits<long>::max()));
    std::vector<std::vector<long>> s(
        n, std::vector<long>(n, std::numeric_limits<long>::max()));
    matrixChainOrder(chainSizes, m, s);
    LLVM_DEBUG(printOptimalParens(s, 1, chain.size() + 1));
    LLVM_DEBUG(llvm::dbgs() << "Min Cost: " << m[1][n - 1] << " \n");

    rebuildChain(chain, operands, s);
  }

  void optimizeChain(linalg::MatmulOp mulOp) {
    std::queue<Operation *> frontier;
    frontier.push(mulOp.getOperation());
    SmallVector<Operation *> chain = {mulOp.getOperation()};

    while (!frontier.empty()) {
      Operation *currOp = frontier.front();
      frontier.pop();

      linalg::MatmulOp currMulOp = dyn_cast_or_null<linalg::MatmulOp>(currOp);
      if (!currMulOp)
        return;
      Value out = currMulOp.outputs()[0];
      for (Operation *user : out.getUsers()) {
        if (user == currOp)
          continue;
        if (isLeftOperand(out, user)) {
          visited.insert(user);
          frontier.push(user);
          chain.push_back(user);
        }
      }
    }

    // no chain detected, bail out.
    if (chain.size() == 1)
      return;

    // let's optmize.
    optimizeChainImpl(chain);

    for (int i = chain.size() - 1; i >= 0; i--)
      toErase.insert(chain[i]);
  }

  void runOnOperation() override {

    getOperation().walk([&](linalg::MatmulOp mulOp) {
      if (visited.count(mulOp.getOperation()))
        return WalkResult::advance();
      visited.insert(mulOp.getOperation());
      optimizeChain(mulOp);
      return WalkResult::advance();
    });

    // TODO: Check if this is safe.
    for (Operation *op : toErase)
      op->erase();
  }

private:
  DenseSet<Operation *> visited;
  DenseSet<Operation *> toErase;
};

} // namespace

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgChainPass() {
  return std::make_unique<LinalgChainOptPass>();
}
