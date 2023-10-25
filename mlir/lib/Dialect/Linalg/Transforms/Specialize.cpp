//===- Specialize.cpp - linalg generic ops to named ops  ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a method to specialize generic operations to named
// operations. Conceptually it is the opposite of generalize.cpp.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-specialization"

using namespace mlir;
using namespace mlir::linalg;

static bool isaCopyOp(GenericOp genericOp) {
  // Structural.
  if (genericOp.getNumParallelLoops() != genericOp.getNumLoops())
    return false;

  // Operands and maps.
  if (genericOp.getNumDpsInputs() != 1 || genericOp.getNumDpsInits() != 1)
    return false;
  auto mapRange = genericOp.getIndexingMapsArray();
  if (mapRange.size() != 2 || !mapRange.front().isIdentity() ||
      !mapRange.back().isIdentity()) {
    return false;
  }

  // Region.
  Region &reg = genericOp.getRegion();
  if (!llvm::hasSingleElement(reg))
    return false;
  return std::distance(reg.front().begin(), reg.front().end()) == 1;
}

FailureOr<LinalgOp> mlir::linalg::specializeGenericOp(RewriterBase &rewriter,
                                                      GenericOp genericOp) {
  if (isaCopyOp(genericOp)) {
    LinalgOp namedOp = rewriter.replaceOpWithNewOp<CopyOp>(
        genericOp, genericOp.getDpsInputs()[0], genericOp.getDpsInits()[0]);
    return namedOp;
  }
  return failure();
}
