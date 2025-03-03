// RUN: mlir-opt %s --log-actions-to=- --test-composite-fixed-point-pass -split-input-file | FileCheck %s
// RUN: mlir-opt %s --log-actions-to=- --composite-fixed-point-pass='name=TestCompositePass pipeline=any(canonicalize,cse)' -split-input-file | FileCheck %s

// CHECK-LABEL: running `TestCompositePass`
//       CHECK: running `Canonicalizer`
//       CHECK: running `CSEPass`
//   CHECK-NOT: running `Canonicalizer`
//   CHECK-NOT: running `CSEPass`
func.func @test() {
  return
}

// -----

// CHECK-LABEL: running `TestCompositePass`
//       CHECK: running `Canonicalizer`
//       CHECK: running `CSEPass`
//       CHECK: running `Canonicalizer`
//       CHECK: running `CSEPass`
//   CHECK-NOT: running `Canonicalizer`
//   CHECK-NOT: running `CSEPass`
func.func @test() {
// this constant will be canonicalized away, causing another pass iteration
  %0 = arith.constant 1.5 : f32
  return
}
