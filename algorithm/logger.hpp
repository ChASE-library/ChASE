// This file is a part of ChASE.
// Copyright (c) 2015-2026, Simulation and Data Laboratory Quantum Materials,
//   Forschungszentrum Juelich GmbH, Germany. All rights reserved.
// License is 3-clause BSD:
// https://github.com/ChASE-library/ChASE

#ifndef CHASE_ALGORITHM_LOGGER_HPP
#define CHASE_ALGORITHM_LOGGER_HPP

#include <cctype>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_set>
#include <vector>

namespace chase
{

/**
 * Canonical log categories. Use these with Log(level, category, msg, rank)
 * and CHASE_LOG_CATEGORIES so users can filter by source.
 *
 * - algorithm: Solver flow (iteration, bounds, locking, degrees, convergence).
 *   All Output() from the algorithm and backends.
 * - performance: Timing and FLOPs (ChasePerfData::print() table).
 * - linalg: Kernel-level (CholQR degree/condition, Rayleigh-Ritz, Lanczos
 *   bounds/NaN). Used when linalg CHASE_OUTPUT is routed through the logger.
 * - interface: API/config (config dump, example output from C/Fortran).
 */

/** Log level for ChASE output. Higher ordinal = more verbose. */
enum class LogLevel
{
    Error = 0,
    Warn = 1,
    Info = 2,
    Debug = 3,
    Trace = 4
};

/** Parse CHASE_LOG_LEVEL env: error, warn, info, debug, trace (case-insensitive). */
inline LogLevel LogLevelFromEnv()
{
    const char* e = std::getenv("CHASE_LOG_LEVEL");
    if (!e)
        return LogLevel::Warn;
    std::string s(e);
    for (auto& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    if (s == "error")
        return LogLevel::Error;
    if (s == "warn" || s == "warning")
        return LogLevel::Warn;
    if (s == "info")
        return LogLevel::Info;
    if (s == "debug")
        return LogLevel::Debug;
    if (s == "trace")
        return LogLevel::Trace;
    return LogLevel::Info;
}

/** Parse CHASE_LOG_RANK env: -1 = all ranks, else single rank. */
inline int LogRankFromEnv()
{
    const char* e = std::getenv("CHASE_LOG_RANK");
    if (!e)
        return 0;
    return std::atoi(e);
}

/** Parse CHASE_LOG_CATEGORIES env: comma-separated (e.g. "algorithm,performance"). Empty/unset = all categories allowed. */
inline std::unordered_set<std::string> LogCategoriesFromEnv()
{
    std::unordered_set<std::string> out;
    const char* e = std::getenv("CHASE_LOG_CATEGORIES");
    if (!e || e[0] == '\0')
        return out;
    std::string s(e);
    std::size_t pos = 0;
    while (pos < s.size())
    {
        std::size_t next = s.find(',', pos);
        std::string cat = (next == std::string::npos) ? s.substr(pos) : s.substr(pos, next - pos);
        pos = (next == std::string::npos) ? s.size() : next + 1;
        while (!cat.empty() && cat.front() == ' ')
            cat.erase(0, 1);
        while (!cat.empty() && cat.back() == ' ')
            cat.pop_back();
        if (!cat.empty())
            out.insert(cat);
    }
    return out;
}

/**
 * Central logger for ChASE: level-based, rank-aware, and category-aware.
 * Default: only rank 0, Info level, all categories. Configure via SetLevel,
 * SetRankFilter, SetCategoryFilter, or env CHASE_LOG_LEVEL, CHASE_LOG_RANK,
 * CHASE_LOG_CATEGORIES.
 */
class ChaseLogger
{
public:
    static ChaseLogger& Instance()
    {
        static ChaseLogger inst;
        return inst;
    }

    void SetLevel(LogLevel level) { level_ = level; }

    LogLevel GetLevel() const { return level_; }

    /** -1 = emit from all ranks; otherwise only from this rank. */
    void SetRankFilter(int rank) { rank_filter_ = rank; }

    int GetRankFilter() const { return rank_filter_; }

    /**
     * Restrict logging to these categories. Empty = allow all categories.
     * E.g. SetCategoryFilter({"algorithm", "performance"}).
     */
    void SetCategoryFilter(std::vector<std::string> const& categories)
    {
        category_filter_.clear();
        for (const auto& c : categories)
            if (!c.empty())
                category_filter_.insert(c);
    }

    /** True if no category filter is set (all allowed) or category is in the allowed set. */
    bool IsCategoryEnabled(const char* category) const
    {
        if (category_filter_.empty())
            return true;
        return category_filter_.count(std::string(category)) != 0;
    }

    /** Initialize from environment (CHASE_LOG_LEVEL, CHASE_LOG_RANK, CHASE_LOG_CATEGORIES). */
    void InitFromEnv()
    {
        level_ = LogLevelFromEnv();
        rank_filter_ = LogRankFromEnv();
        category_filter_ = LogCategoriesFromEnv();
    }

    /**
     * Log a message if it passes all three filters (level, rank, category).
     * Printed only when: level <= threshold AND rank matches filter AND category is enabled.
     * Level and category are independent: e.g. SetLevel(Info) + SetCategoryFilter({"algorithm"})
     * shows algorithm messages at Info and above, and no performance messages regardless of level.
     */
    void Log(LogLevel level, const char* category, const std::string& msg,
             int rank) const
    {
#ifdef CHASE_OUTPUT
        if (static_cast<int>(level) > static_cast<int>(level_))
            return;
        if (rank_filter_ >= 0 && rank != rank_filter_)
            return;
        if (!IsCategoryEnabled(category))
            return;
        std::cout << msg;
        std::cout.flush();
#else
        (void)level;
        (void)category;
        (void)msg;
        (void)rank;
#endif
    }

private:
    ChaseLogger()
        : level_(LogLevelFromEnv())
        , rank_filter_(LogRankFromEnv())
        , category_filter_(LogCategoriesFromEnv())
    {
    }

    LogLevel level_;
    int rank_filter_;
    std::unordered_set<std::string> category_filter_; // empty = all categories allowed
};

inline ChaseLogger& GetLogger() { return ChaseLogger::Instance(); }

} // namespace chase

#endif
