// RUN: mlir-opt -test-transform-dialect-interpreter %s -allow-unregistered-dialect -verify-diagnostics --split-input-file | FileCheck %s

// CHECK: func.func @foo() {
// CHECK:   "dummy_op"() : () -> ()
// CHECK: }
// CHECK-NOT: func.func @bar
func.func @bar() {
  "another_op"() : () -> ()
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["func.func"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.replace %0 {
    func.func @foo() {
      "dummy_op"() : () -> ()
    }
  } : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @bar(%arg0: i1) {
  "another_op"(%arg0) : (i1) -> ()
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["another_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  // expected-error @+1 {{expected target with 0 operands}}
  transform.structured.replace %0 {
    "dummy_op"() : () -> ()
  } : (!transform.any_op) -> !transform.any_op
}

// -----

func.func @bar() {
  "another_op"() : () -> ()
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["another_op"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  transform.structured.replace %0 {
  ^bb0(%a: i1):
    // expected-error @+1 {{expected replacement with 0 operands}}
    "dummy_op"(%a) : (i1) -> ()
  } : (!transform.any_op) -> !transform.any_op
}

// -----

// CHECK-LABEL: main
// CHECK-NOT: arith.addi
// CHECK: %[[C2:.+]] = arith.constant 2 : i32
// CHECK-NEXT: %[[C3:.+]] = arith.constant 3 : i32
// CHECK-NEXT: %[[MUL:.+]] = arith.muli %[[C2]], %[[C3]] : i32
// CHECK-NEXT: return %[[MUL]] : i32

func.func @main() -> i32 {
  %0 = arith.constant 2 : i32
  %1 = arith.constant 3 : i32
  %2 = arith.addi %0, %1 : i32
  return %2 : i32
}

transform.sequence failures(propagate) {
^bb1(%arg1: !transform.any_op):
  %0 = transform.structured.match ops{["arith.addi"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %csts = transform.structured.match ops{["arith.constant"]} in %arg1 : (!transform.any_op) -> !transform.any_op
  %split_csts:2 = split_handle %csts : (!transform.any_op) -> (!transform.any_op, !transform.any_op)
  %c2 = transform.get_result %split_csts#0[0] : (!transform.any_op) -> !transform.any_value
  %c3 = transform.get_result %split_csts#1[0] : (!transform.any_op) -> !transform.any_value

  transform.structured.replace %0 (%c2, %c3 : !transform.any_value, !transform.any_value) {
  ^bb0(%op1: i32, %op2: i32):
    %3 = arith.muli %op1, %op2 : i32
  } : (!transform.any_op) -> !transform.any_op
}
