// RUN: mlir-opt --transform-interpreter --split-input-file --verify-diagnostics %s | FileCheck %s

#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1) -> (d0)>
#map2 = affine_map<(d0, d1) -> (d1, d0)>

func.func @broadcast_copy_expect_no_match(%arg0: memref<?xf32>, %arg1: memref<?x?xf32>) {
  // expected-note @below {{when applied to this op}}
  linalg.generic {
    indexing_maps = [#map1, #map], 
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<?xf32>) outs(%arg1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

func.func @not_a_copy_expect_no_match(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
  // expected-note @below {{when applied to this op}}
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]}
    ins(%arg0 : memref<?x?xf32>) outs(%arg1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      %0 = arith.addf %in, %out : f32
      linalg.yield %0 : f32
  }
  return
}

func.func @transpose_op_expect_no_match(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
  // expected-note @below {{when applied to this op}}
  linalg.generic {
    indexing_maps = [#map, #map2], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<?x?xf32>) outs(%arg1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    // expected-error @below {{failed to apply}}
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}

// -----

#map = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL: generalize_trivial_copy
func.func @generalize_trivial_copy(%arg0: memref<?x?xf32>, %arg1: memref<?x?xf32>) {
  // CHECK: linalg.copy
  // CHECK-NOT: linalg.generic
  linalg.generic {
    indexing_maps = [#map, #map], 
    iterator_types = ["parallel", "parallel"]} 
    ins(%arg0 : memref<?x?xf32>) outs(%arg1 : memref<?x?xf32>) {
    ^bb0(%in: f32, %out: f32):
      linalg.yield %in : f32
  }
  return
}

module attributes {transform.with_named_sequence} {
  transform.named_sequence @__transform_main(%arg1: !transform.any_op {transform.readonly}) {
    %0 = transform.structured.match interface{LinalgOp} in %arg1 : (!transform.any_op) -> !transform.any_op
    %1 = transform.structured.specialize %0 : (!transform.any_op) -> !transform.any_op
    transform.yield
  }
}
