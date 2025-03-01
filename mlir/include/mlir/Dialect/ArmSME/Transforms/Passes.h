//===- Passes.h - Pass Entrypoints ------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/ArmSME/Transforms/PassesEnums.h.inc"
#include "mlir/Pass/Pass.h"

namespace mlir {

class RewritePatternSet;

namespace arm_sme {
//===----------------------------------------------------------------------===//
// The EnableArmStreaming pass.
//===----------------------------------------------------------------------===//
#define GEN_PASS_DECL
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"

//===----------------------------------------------------------------------===//
// Registration
//===----------------------------------------------------------------------===//

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "mlir/Dialect/ArmSME/Transforms/Passes.h.inc"

} // namespace arm_sme
} // namespace mlir

#endif // MLIR_DIALECT_ARMSME_TRANSFORMS_PASSES_H
