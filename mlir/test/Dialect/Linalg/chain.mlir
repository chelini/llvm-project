// RUN: mlir-opt %s --linalg-chain-opt | FileCheck %s
module {
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
