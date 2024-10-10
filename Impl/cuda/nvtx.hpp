#pragma once

#include "nvtx3/nvToolsExt.h"

#ifdef USE_NVTX
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


