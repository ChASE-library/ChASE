// This file is a part of ChASE.
// Copyright (c) 2015-2024, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#pragma once

#ifdef USE_NVTX
#include "nvtx3/nvToolsExt.h"

/**
 * @brief RAII-based class for managing NVTX (NVIDIA Tools Extension) ranges.
 *
 * This class provides a simple way to profile sections of code using NVTX,
 which is useful
 * for visualizing code execution in NVIDIA's Nsight and other profiling tools.
 The `ScopedNvtxRange`
 * class uses the RAII (Resource Acquisition Is Initialization) idiom to
 automatically begin and end
 * an NVTX range within a scope.
 *
 * When an object of `ScopedNvtxRange` is created, it pushes an NVTX range with
 the given name.
 * When the object goes out of scope, the destructor automatically pops the NVTX
 range, ensuring
 * proper profiling of the enclosed code section.
 *
 * This class is especially useful for profiling CUDA or GPU-bound code,
 allowing developers
 * to identify bottlenecks in their code execution.

 * @note This class requires NVIDIA’s NVTX library and a compatible profiler
 (e.g., Nsight).
 */
class ScopedNvtxRange
{
public:
    /**
     * @brief Constructor that begins an NVTX range.
     *
     * The constructor pushes an NVTX range with the specified name. This range
     * will be active until the object goes out of scope and the destructor is
     * called.
     *
     * @param name The name of the NVTX range to be pushed. Typically, this is
     * the function name or a description of the code block being profiled.
     */
    ScopedNvtxRange(const char* name) { nvtxRangePushA(name); }
    /**
     * @brief Destructor that ends the NVTX range.
     *
     * The destructor automatically pops the NVTX range when the
     * `ScopedNvtxRange` object goes out of scope. This ensures that the range
     * is correctly closed.
     */
    ~ScopedNvtxRange() { nvtxRangePop(); }
};
/**
 * @brief Macro to create a scoped NVTX range for the current function.
 *
 * This macro creates an instance of `ScopedNvtxRange` with the name of the
 * current function
 * (`__PRETTY_FUNCTION__`). It is a convenient way to automatically profile
 * entire functions.
 *
 * Usage example:
 * @code
 * SCOPED_NVTX_RANGE();
 * // code to profile
 * @endcode
 *
 * @note This macro is only active if the NVTX library is available. If it is
 * not available, the macro does nothing.
 */
#define SCOPED_NVTX_RANGE() ScopedNvtxRange scopedNvtxRange(__PRETTY_FUNCTION__)
#else
#define SCOPED_NVTX_RANGE()
#endif
