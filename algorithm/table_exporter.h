/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#pragma once

#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "configuration.h"
#include "performance.h"
#include "types.h"

extern double CHASE_ADJUST_LOWERB;

namespace chase {

template <typename T>
void export_sql(std::string name, std::size_t idx, ChaseConfig<T> conf,
                ChasePerfData perf) {
  std::vector<std::string> names;
  std::vector<std::string> values;

  names.push_back("version");
  values.push_back(std::to_string(CHASE_VERSION_MAJOR) + "." +
                   std::to_string(CHASE_VERSION_MINOR));

  int size = 1;
#ifdef HAS_MPI
  MPI_Comm_size(MPI_COMM_WORLD, &size);
#endif
  names.push_back("size");
  values.push_back(std::to_string(size));

  names.push_back("iterations");
  values.push_back(std::to_string(perf.get_iter_count()));

  names.push_back("filtered_vecs");
  values.push_back(std::to_string(perf.get_filtered_vecs()));

  names.push_back("name");
  values.push_back("'" + name + "'");

  names.push_back("idx");
  values.push_back(std::to_string(idx));

  names.push_back("nev");
  values.push_back(std::to_string(conf.getNev()));

  names.push_back("nex");
  values.push_back(std::to_string(conf.getNex()));

  names.push_back("N");
  values.push_back(std::to_string(conf.getN()));

  names.push_back("deg");
  values.push_back(std::to_string(conf.getDeg()));

  names.push_back("tol");
  values.push_back(std::to_string(conf.getTol()));

  names.push_back("approx");
  values.push_back(std::to_string(conf.use_approx()));

  names.push_back("opt");
  values.push_back(std::to_string(conf.do_optimization()));

  names.push_back("chase_adjust_lowerb");
  values.push_back(std::to_string(CHASE_ADJUST_LOWERB));


  auto timings = perf.get_timings();
  names.push_back("time_all");
  values.push_back(
      std::to_string(timings[ChasePerfData::TimePtrs::All].count()));

  // todo other timings

  /*
  std::cout << "INSERT INTO TABLE experiments (version, size, iteration, "
               "filtered_vecs, time_all, time_lanczos, time_degrees, "
               "time_filter, time_qr, time_rr, time_locking) VALUES ("
            << CHASE_VERSION_MAJOR << "." << CHASE_VERSION_MINOR << ","
            << size << "," << chase_iteration_count << ","
            << chase_filtered_vecs << "," << timings[TimePtrs::All].count()
            << " , " << timings[TimePtrs::Lanczos].count() << " , "
            << timings[TimePtrs::Degrees].count() << " , "
            << timings[TimePtrs::Filter].count() << " , "
            << timings[TimePtrs::Qr].count() << " , "
            << timings[TimePtrs::Rr].count() << " , "
            << timings[TimePtrs::Resids_Locking].count() << " ); "
            << std::endl;
  */

  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y %H-%M-%S");
  auto now = oss.str();

  assert(names.size() == values.size());
  std::cout << "INSERT INTO chase_experiments (";
  for (auto name : names) std::cout << name << ",";
  std::cout << " now) VALUES (";
  for (auto value : values) std::cout << value << ",";
  std::cout << "'" + now + "' );\n";
}
} // namespace chase
