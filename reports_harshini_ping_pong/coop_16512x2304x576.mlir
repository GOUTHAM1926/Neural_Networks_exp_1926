module {
  func.func @main(%arg0: memref<16512x2304xbf16, 1> {nova.arg_kind = "input"}, %arg1: memref<2304x576xbf16, 1> {nova.arg_kind = "input"}, %arg2: memref<16512x576xbf16, 1> {bufferization.writable = true}) attributes {llvm.emit_c_interface} {
    %0 = bufferization.to_tensor %arg0 restrict : memref<16512x2304xbf16, 1> to tensor<16512x2304xbf16>
    %1 = bufferization.to_tensor %arg1 restrict : memref<2304x576xbf16, 1> to tensor<2304x576xbf16>
    %2 = nova.matmul %0, %1 : tensor<16512x2304xbf16>, tensor<2304x576xbf16>
    bufferization.materialize_in_destination %2 in writable %arg2 : (tensor<16512x576xbf16>, memref<16512x576xbf16, 1>) -> ()
    return
  }
}
