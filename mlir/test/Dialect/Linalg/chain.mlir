// RUN: mlir-opt %s -split-input-file -linalg-comprehensive-module-bufferize -linalg-chain-opt | FileCheck %s
module {
  // CHECK: Optimized
  // CHECK: moduleFn1
  func private @moduleFn1(%arg0: memref<30x15xf32>, %arg1: memref<35x15xf32>, 
                          %arg2: memref<30x35xf32>, %arg3: memref<30x5xf32>, 
                          %arg4: memref<15x5xf32>, %arg5: memref<30x10xf32>, 
                          %arg6: memref<5x10xf32>, %arg7: memref<30x20xf32>, 
                          %arg8: memref<10x20xf32>, %arg9: memref<30x25xf32>, 
                          %arg10: memref<20x25xf32>) attributes {llvm.emit_c_interface} {
    linalg.matmul ins(%arg2, %arg1 : memref<30x35xf32>, memref<35x15xf32>) outs(%arg0 : memref<30x15xf32>)
    linalg.matmul ins(%arg0, %arg4 : memref<30x15xf32>, memref<15x5xf32>) outs(%arg3 : memref<30x5xf32>)
    linalg.matmul ins(%arg3, %arg6 : memref<30x5xf32>, memref<5x10xf32>) outs(%arg5 : memref<30x10xf32>)
    linalg.matmul ins(%arg5, %arg8 : memref<30x10xf32>, memref<10x20xf32>) outs(%arg7 : memref<30x20xf32>)
    linalg.matmul ins(%arg7, %arg10 : memref<30x20xf32>, memref<20x25xf32>) outs(%arg9 : memref<30x25xf32>)
    return
  }
}

// -----

module {
  // CHECK: Optimized
  // CHECK: moduleFn1
  func private @moduleFn1(%arg0: tensor<30x15xf32>, %arg1: tensor<35x15xf32>,
                          %arg2: tensor<30x35xf32>, %arg3: tensor<30x5xf32>,
                          %arg4: tensor<15x5xf32>, %arg5: tensor<30x10xf32>,
                          %arg6: tensor<5x10xf32>, %arg7: tensor<30x20xf32>, 
                          %arg8: tensor<10x20xf32>, %arg9: tensor<30x25xf32>, 
                          %arg10: tensor<20x25xf32>) -> tensor<30x10xf32> attributes {llvm.emit_c_interface} {
    %0 = linalg.matmul ins(%arg2, %arg1 : tensor<30x35xf32>, tensor<35x15xf32>) outs(%arg0: tensor<30x15xf32>) -> tensor<30x15xf32>
    %1 = linalg.matmul ins(%0, %arg4 : tensor<30x15xf32>, tensor<15x5xf32>) outs(%arg3 : tensor<30x5xf32>) -> tensor<30x5xf32>
    %2 = linalg.matmul ins(%1, %arg6 : tensor<30x5xf32>, tensor<5x10xf32>) outs(%arg5 : tensor<30x10xf32>) -> tensor<30x10xf32>
    return %2 : tensor<30x10xf32>
  }
}

// -----

module {
  // CHECK: moduleFn1
  func private @moduleFn1(%arg0: tensor<30x15xf32>, %arg1: tensor<35x15xf32>, %arg2: tensor<30x35xf32>, %arg3: tensor<30x5xf32>, %arg4: tensor<15x5xf32>, %arg5: tensor<30x10xf32>, %arg6: tensor<5x10xf32>, %arg7: tensor<30x20xf32>, %arg8: tensor<10x20xf32>, %arg9: tensor<30x25xf32>, %arg10: tensor<20x25xf32>) -> tensor<30x25xf32> attributes {llvm.emit_c_interface} {
    %0 = call @mxm2(%arg2, %arg1, %arg0) : (tensor<30x35xf32>, tensor<35x15xf32>, tensor<30x15xf32>) -> tensor<30x15xf32>
    %1 = call @mxm3(%0, %arg4, %arg3) : (tensor<30x15xf32>, tensor<15x5xf32>, tensor<30x5xf32>) -> tensor<30x5xf32>
    %2 = call @mxm4(%1, %arg6, %arg5) : (tensor<30x5xf32>, tensor<5x10xf32>, tensor<30x10xf32>) -> tensor<30x10xf32>
    %3 = call @mxm5(%2, %arg8, %arg7) : (tensor<30x10xf32>, tensor<10x20xf32>, tensor<30x20xf32>) -> tensor<30x20xf32>
    %4 = call @mxm6(%3, %arg10, %arg9) : (tensor<30x20xf32>, tensor<20x25xf32>, tensor<30x25xf32>) -> tensor<30x25xf32>
    return %4 : tensor<30x25xf32>
  }
  func private @mxm2(%arg0: tensor<30x35xf32>, %arg1: tensor<35x15xf32>, %arg2: tensor<30x15xf32>) -> tensor<30x15xf32> attributes {llvm.emit_c_interface} {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<30x35xf32>, tensor<35x15xf32>) outs(%arg2 : tensor<30x15xf32>) -> tensor<30x15xf32>
    return %0 : tensor<30x15xf32>
  }
  func private @mxm3(%arg0: tensor<30x15xf32>, %arg1: tensor<15x5xf32>, %arg2: tensor<30x5xf32>) -> tensor<30x5xf32> attributes {llvm.emit_c_interface} {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<30x15xf32>, tensor<15x5xf32>) outs(%arg2 : tensor<30x5xf32>) -> tensor<30x5xf32>
    return %0 : tensor<30x5xf32>
  }
  func private @mxm4(%arg0: tensor<30x5xf32>, %arg1: tensor<5x10xf32>, %arg2: tensor<30x10xf32>) -> tensor<30x10xf32> attributes {llvm.emit_c_interface} {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<30x5xf32>, tensor<5x10xf32>) outs(%arg2 : tensor<30x10xf32>) -> tensor<30x10xf32>
    return %0 : tensor<30x10xf32>
  }
  func private @mxm5(%arg0: tensor<30x10xf32>, %arg1: tensor<10x20xf32>, %arg2: tensor<30x20xf32>) -> tensor<30x20xf32> attributes {llvm.emit_c_interface} {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<30x10xf32>, tensor<10x20xf32>) outs(%arg2 : tensor<30x20xf32>) -> tensor<30x20xf32>
    return %0 : tensor<30x20xf32>
  }
  func private @mxm6(%arg0: tensor<30x20xf32>, %arg1: tensor<20x25xf32>, %arg2: tensor<30x25xf32>) -> tensor<30x25xf32> attributes {llvm.emit_c_interface} {
    %0 = linalg.matmul ins(%arg0, %arg1 : tensor<30x20xf32>, tensor<20x25xf32>) outs(%arg2 : tensor<30x25xf32>) -> tensor<30x25xf32>
    return %0 : tensor<30x25xf32>
  }
}
