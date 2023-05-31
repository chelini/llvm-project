// RUN: mlir-opt -allow-unregistered-dialect %s -affine-loop-unroll="use-attribute unroll-full" -split-input-file | FileCheck %s 

// CHECK-LABEL: func @loop_nest_simplest() {
func.func @loop_nest_simplest() {
  // CHECK-COUNT-4: arith.constant
  affine.for %i = 0 to 100 step 2 {
    affine.for %j = 0 to 4 {
      %x = arith.constant 1 : i32
    } {"unroll_me"}
  }
  return 
}

// -----

// CHECK-LABEL: func @loop_nest_simplest() {
func.func @loop_nest_simplest() {
  // CHECK-COUNT-1: arith.constant
  affine.for %i = 0 to 100 step 2 {
    affine.for %j = 0 to 4 {
      %x = arith.constant 1 : i32
    }
  } 
  return
}

