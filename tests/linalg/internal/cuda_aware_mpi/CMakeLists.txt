setup_test(HEMMGPUDistTest hemm.cpp LIBRARIES pchase_gpu)
setup_test(CholQRGPUDistTest cholqr.cpp LIBRARIES pchase_gpu)
setup_test(LanczosGPUDistTest lanczos.cpp LIBRARIES pchase_gpu)
setup_test(RayleighRitzGPUDistTest rayleighRitz.cpp LIBRARIES pchase_gpu)
setup_test(ResidualsGPUDistTest residuals.cpp LIBRARIES pchase_gpu)
setup_test(shiftDiagonalGPUDistTest shiftDiagonal.cpp LIBRARIES pchase_gpu)
setup_test(symOrHermGPU2DistTest symOrHerm.cpp LIBRARIES pchase_gpu)