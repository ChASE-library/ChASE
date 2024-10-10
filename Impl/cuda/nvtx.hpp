#pragma once

#ifdef USE_NVTX
#include "nvtx3/nvToolsExt.h"

class ScopedNvtxRange {
public:
    ScopedNvtxRange(const char* name) {
        nvtxRangePushA(name);
    }
    ~ScopedNvtxRange() {
        nvtxRangePop();
    }
};
#define SCOPED_NVTX_RANGE() ScopedNvtxRange scopedNvtxRange(__PRETTY_FUNCTION__)
#else
#define SCOPED_NVTX_RANGE()
#endif


