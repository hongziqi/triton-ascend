// RUN: triton-opt --triton-to-linalg --split-input-file %s | FileCheck %s
//
// tt.addptr base variants (rewriteAddPtr + materializeIntToPtrAsMemref):
//   1) int_to_ptr
//   2) int_to_ptr + bitcast
//   3) int_to_ptr + 2*bitcast
//   4) bitcast(arg)
//   5) 2*bitcast(arg)
//
// Note: tt.bitcast requires equal bitwidth (i32 <-> f32).

// CHECK-LABEL: func.func @addptr_int_to_ptr
// CHECK: %[[PC:.*]] = hivm.hir.pointer_cast
// CHECK: memref.reinterpret_cast %[[PC]]
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @addptr_int_to_ptr(
      %arg0: i64 {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %0 = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %2 = tt.splat %0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %4 = tt.load %3 : tensor<256x!tt.ptr<f32>>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %6 = tt.addptr %5, %1 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %6, %4 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @addptr_int_to_ptr_bitcast
// CHECK: %[[PC:.*]] = hivm.hir.pointer_cast
// CHECK-SAME: memref<?xf32>
// CHECK: memref.reinterpret_cast %[[PC]]
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @addptr_int_to_ptr_bitcast(
      %arg0: i64 {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %0 = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<i32>
    %1 = tt.bitcast %0 : !tt.ptr<i32> -> !tt.ptr<f32>
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %5 = tt.load %4 : tensor<256x!tt.ptr<f32>>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %7 = tt.addptr %6, %2 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %7, %5 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @addptr_int_to_ptr_bitcast2
// CHECK: %[[PC:.*]] = hivm.hir.pointer_cast
// CHECK-SAME: memref<?xf32>
// CHECK: memref.reinterpret_cast %[[PC]]
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @addptr_int_to_ptr_bitcast2(
      %arg0: i64 {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %0 = tt.int_to_ptr %arg0 : i64 -> !tt.ptr<f32>
    %1 = tt.bitcast %0 : !tt.ptr<f32> -> !tt.ptr<i32>
    %2 = tt.bitcast %1 : !tt.ptr<i32> -> !tt.ptr<f32>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %6 = tt.load %5 : tensor<256x!tt.ptr<f32>>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %8 = tt.addptr %7, %3 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %8, %6 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @addptr_bitcast
// CHECK-NOT: hivm.hir.pointer_cast
// CHECK: unrealized_conversion_cast
// CHECK: memref.reinterpret_cast
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @addptr_bitcast(
      %arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %0 = tt.bitcast %arg0 : !tt.ptr<i32> -> !tt.ptr<f32>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %2 = tt.splat %0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %3 = tt.addptr %2, %1 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %4 = tt.load %3 : tensor<256x!tt.ptr<f32>>
    %5 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %6 = tt.addptr %5, %1 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %6, %4 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: func.func @addptr_bitcast2
// CHECK-NOT: hivm.hir.pointer_cast
// CHECK: memref.reinterpret_cast
// CHECK: memref.copy
// CHECK: return
module {
  tt.func public @addptr_bitcast2(
      %arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32},
      %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}
  ) attributes {noinline = false} {
    %0 = tt.bitcast %arg0 : !tt.ptr<f32> -> !tt.ptr<i32>
    %1 = tt.bitcast %0 : !tt.ptr<i32> -> !tt.ptr<f32>
    %2 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %3 = tt.splat %1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %5 = tt.load %4 : tensor<256x!tt.ptr<f32>>
    %6 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %7 = tt.addptr %6, %2 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    tt.store %7, %5 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}
