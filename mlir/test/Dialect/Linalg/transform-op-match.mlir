// RUN: mlir-opt %s --test-transform-dialect-interpreter -allow-unregistered-dialect --split-input-file --verify-diagnostics

func.func @bar() {
  // expected-remark @below {{matched op name}}
  // expected-remark @below {{matched attr name}}
  %0 = arith.constant {my_attr} 0: i32
  // expected-remark @below {{matched op name}}
  %1 = arith.constant 1 : i32
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %match_name = transform.structured.match ops{["arith.constant"]} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.test_print_remark_at_operand %match_name, "matched op name" : !pdl.operation
  transform.test_consume_operand %match_name : !pdl.operation

  %match_attr = transform.structured.match ops{["arith.constant"]} attributes{my_attr} in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.test_print_remark_at_operand %match_attr, "matched attr name" : !pdl.operation
  transform.test_consume_operand %match_attr : !pdl.operation
}

// -----

func.func @by_type() {
  %0 = arith.constant 0: i32
  // expected-remark @below {{matched op name}}
  %1 = arith.constant 1.0 : f32
  return
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %match_name = transform.structured.match
    ops{["arith.constant"]} filter_result_type = f32 in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.test_print_remark_at_operand %match_name, "matched op name" : !pdl.operation
  transform.test_consume_operand %match_name : !pdl.operation
}

// -----

#map0 = affine_map<(d0, d1, d2) -> (d0, d1, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d1, d0, d2)>
func.func @match_complex_attribute(%arg0: tensor<12x128x32xf32>)
    -> tensor<128x12x32xf32> {
  %0 = tensor.empty() : tensor<128x12x32xf32>
  // expected-remark @below {{matched complex attr}}
  %1 = linalg.generic {indexing_maps = [#map0, #map1],
                       iterator_types = ["parallel", "parallel", "parallel"]}
    ins(%arg0 : tensor<12x128x32xf32>)
    outs(%0 : tensor<128x12x32xf32>) {
  ^bb0(%arg1: f32, %arg2: f32):
    linalg.yield %arg1 : f32
  } -> tensor<128x12x32xf32>
  return %1 : tensor<128x12x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %match_attr = transform.structured.match
      ops{["linalg.generic"]}
      attributes{iterator_types = [
        #linalg.iterator_type<parallel>,
        #linalg.iterator_type<parallel>,
        #linalg.iterator_type<parallel>]}
      in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.test_print_remark_at_operand %match_attr, "matched complex attr" : !pdl.operation
  transform.test_consume_operand %match_attr : !pdl.operation

  %no_match = transform.structured.match
      attributes{iterator_types = [
        #linalg.iterator_type<parallel>,
        #linalg.iterator_type<parallel>,
        #linalg.iterator_type<reduction>]}
      in %arg1 : (!pdl.operation) -> !pdl.operation
// expected-remark @below {{0}}
  transform.test_print_number_of_associated_payload_ir_ops %no_match
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d5, d2 + d6, d3 + d7, d8)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d1, d5, d6, d7, d8, d4)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7, d8) -> (d0, d1, d2, d3, d4)>

func.func @match_blocked_conv(%pack: tensor<1x2x56x56x32xf32>, 
  %pack_0: tensor<2x2x1x1x32x32xf32>, 
  %pack_1: tensor<1x2x56x56x32xf32>) -> tensor<1x2x56x56x32xf32> {
  // expected-remark @below {{matched blocked convolution}}
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", 
                      "reduction", "reduction", "reduction", "reduction"]} 
    ins(%pack, %pack_0 : tensor<1x2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>) 
    outs(%pack_1 : tensor<1x2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_2 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<1x2x56x56x32xf32>
  return %0 : tensor<1x2x56x56x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %match_name = transform.structured.match interface{BlockedConvolution}
    in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.test_print_remark_at_operand %match_name, "matched blocked convolution" 
    : !pdl.operation
  transform.test_consume_operand %match_name : !pdl.operation
}

// -----

#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d4, d1 + d5, d2 + d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d4, d5, d6, d7, d3)>
#map2 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3)>

func.func @match_blocked_conv(%pack: tensor<2x56x56x32xf32>,
  %pack_0: tensor<2x2x1x1x32x32xf32>,
  %pack_1: tensor<2x56x56x32xf32>) -> tensor<2x56x56x32xf32> {
  // expected-remark @below {{matched blocked convolution}}
  %0 = linalg.generic {
    indexing_maps = [#map, #map1, #map2], 
    iterator_types = ["parallel", "parallel", "parallel", "parallel",
                      "reduction", "reduction", "reduction", "reduction"]} 
    ins(%pack, %pack_0 : tensor<2x56x56x32xf32>, tensor<2x2x1x1x32x32xf32>)
    outs(%pack_1 : tensor<2x56x56x32xf32>) {
    ^bb0(%in: f32, %in_2: f32, %out: f32):
      %1 = arith.mulf %in, %in_2 : f32
      %2 = arith.addf %out, %1 : f32
      linalg.yield %2 : f32
  } -> tensor<2x56x56x32xf32>
  return %0 : tensor<2x56x56x32xf32>
}

transform.sequence failures(propagate) {
^bb1(%arg1: !pdl.operation):
  %match_name = transform.structured.match interface{BlockedConvolution}
    in %arg1 : (!pdl.operation) -> !pdl.operation
  transform.test_print_remark_at_operand %match_name, "matched blocked convolution" 
    : !pdl.operation
  transform.test_consume_operand %match_name : !pdl.operation
}
