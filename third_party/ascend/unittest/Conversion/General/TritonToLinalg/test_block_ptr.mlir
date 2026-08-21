// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s
//
// make_tensor_ptr base variants:
//   1) int_to_ptr
//   2) int_to_ptr + bitcast
//   3) int_to_ptr + 2*bitcast
//   4) bitcast(arg)
//   5) 2*bitcast(arg)

// CHECK-LABEL: func.func @kernel_int_to_ptr
// CHECK: %[[PC:.*]] = hivm.hir.pointer_cast
// CHECK: memref.reinterpret_cast %[[PC]]
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @kernel_int_to_ptr(
      %arg0: i64 {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
    %1 = tt.make_tensor_ptr %0,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    %2 = tt.load %1 : !tt.ptr<tensor<256xf32>>
    %3 = tt.make_tensor_ptr %arg1,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    tt.store %3, %2 : !tt.ptr<tensor<256xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @kernel_int_to_ptr_bitcast
// CHECK: %[[PC:.*]] = hivm.hir.pointer_cast
// CHECK-SAME: memref<?xf32>
// CHECK: memref.reinterpret_cast %[[PC]]
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @kernel_int_to_ptr_bitcast(
      %arg0: i64 {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<i32>
    %1 = tt.bitcast %0 : !tt.ptr<i32> -> !tt.ptr<f32>
    %2 = tt.make_tensor_ptr %1,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    %3 = tt.load %2 : !tt.ptr<tensor<256xf32>>
    %4 = tt.make_tensor_ptr %arg1,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    tt.store %4, %3 : !tt.ptr<tensor<256xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @kernel_int_to_ptr_bitcast2
// CHECK: %[[PC:.*]] = hivm.hir.pointer_cast
// CHECK-SAME: memref<?xf32>
// CHECK: memref.reinterpret_cast %[[PC]]
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @kernel_int_to_ptr_bitcast2(
      %arg0: i64 {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
    %1 = tt.bitcast %0 : !tt.ptr<f32> -> !tt.ptr<i32>
    %2 = tt.bitcast %1 : !tt.ptr<i32> -> !tt.ptr<f32>
    %3 = tt.make_tensor_ptr %2,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    %4 = tt.load %3 : !tt.ptr<tensor<256xf32>>
    %5 = tt.make_tensor_ptr %arg1,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    tt.store %5, %4 : !tt.ptr<tensor<256xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @kernel_bitcast
// CHECK-NOT: hivm.hir.pointer_cast
// CHECK: memref.reinterpret_cast
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @kernel_bitcast(
      %arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.bitcast %arg0 : !tt.ptr<i32> -> !tt.ptr<f32>
    %1 = tt.make_tensor_ptr %0,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    %2 = tt.load %1 : !tt.ptr<tensor<256xf32>>
    %3 = tt.make_tensor_ptr %arg1,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    tt.store %3, %2 : !tt.ptr<tensor<256xf32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @kernel_bitcast2
// CHECK-NOT: hivm.hir.pointer_cast
// CHECK: memref.reinterpret_cast
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @kernel_bitcast2(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %c256_i64 = arith.constant 256 : i64
    %c1_i64 = arith.constant 1 : i64
    %0 = tt.bitcast %arg0 : !tt.ptr<f32> -> !tt.ptr<i32>
    %1 = tt.bitcast %0 : !tt.ptr<i32> -> !tt.ptr<f32>
    %2 = tt.make_tensor_ptr %1,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    %3 = tt.load %2 : !tt.ptr<tensor<256xf32>>
    %4 = tt.make_tensor_ptr %arg1,
         [%c256_i64], [%c1_i64], [%c0_i32] {order = array<i32: 0>}
         : <tensor<256xf32>>
    tt.store %4, %3 : !tt.ptr<tensor<256xf32>>
    tt.return
  }
}
