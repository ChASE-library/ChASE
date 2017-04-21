#ifndef CHASE_TESTRESULT_H
#define CHASE_TESTRESULT_H

// #define CHASE_TESTRESULT_WRITE true
// #define CHASE_TESTRESULT_COMPARE false

#include <unordered_map>
#include <vector>
#include <complex>
#include <iostream>

#include <fstream>
#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>



template<typename T>
void assertEqual( typename std::unordered_map<std::string,T>::iterator it,
                  std::unordered_map<std::string,T> const &map,
                  std::size_t &tests, std::size_t &fails)
{
  auto rhs = map.at( it->first );
  if( it->second != rhs )
  {
    fails++;
    std::cout << it->first << " fails comparison\t"
              << "(calc)" << it->second<< " != " << rhs << "(reference)"
              << std::endl;
  }
  tests++;
}

template<typename T>
void assertEqual( typename std::unordered_map<std::string,T>::iterator it,
                  std::unordered_map<std::string,T> const &map,
                  T tolerance,std::size_t &tests, std::size_t &fails)
{
  auto rhs = map.at( it->first );
  if( std::abs((it->second - rhs)/it->second) > tolerance )
  {
    fails++;
    std::cout << it->first << " fails comparison\t|"
              << it->second<< " - " << rhs << "| > " << tolerance
              << std::endl;
  }
  tests++;
}


class TestResultIteration {
public:
  friend class boost::serialization::access;
  friend std::ostream & operator<<(std::ostream &os, const TestResultIteration &tr);

  TestResultIteration()
    {};

  void compareMembers( TestResultIteration &ref, std::size_t &tests, std::size_t &fails ){
    for( auto it = intMap.begin(); it != intMap.end(); ++it )
      assertEqual( it, ref.intMap, tests, fails );
    for( auto it = doubleMap.begin(); it != doubleMap.end(); ++it )
      assertEqual<double>( it, ref.doubleMap, 1e-6, tests, fails );
  };
  void registerValue( std::string key, std::size_t value )
  {
    intMap.insert({ key, value });
  };
  void registerValue( std::string key, double value )
    {
      doubleMap.insert({ key, value });
    };

private:
  std::unordered_map< std::string, std::size_t > intMap;
  std::unordered_map< std::string, double > doubleMap;

  template<class Archive>
    void serialize(Archive & ar, const std::size_t version)
  {
    ar & BOOST_SERIALIZATION_NVP(intMap)
      & BOOST_SERIALIZATION_NVP(doubleMap);
  }
};


class TestResult {
public:
  friend class boost::serialization::access;
  friend std::ostream & operator<<(std::ostream &os, const TestResult &tr);

  TestResult();

  TestResult( bool compare_, std::string name_ );

  TestResult( bool compare, std::string name,
              std::size_t n, std::size_t nev, std::size_t nex, std::size_t deg,
              double tol, char mode, char opt, bool sequence);

  std::string name();

  template<typename T>
  void registerValue( std::size_t iteration, std::string key, T value )
  {
    // ensure the vector is large enough
    iterationResults.resize( std::max( iteration, iterationResults.size() ) );
    iterationResults[iteration-1].registerValue( key, value );
  }


  void done();

private:

  bool compare;
  std::string fileName;
  std::vector<TestResultIteration> iterationResults;

  std::string path_in, path_eigp;
  void set_path_in( std::string path_in );
  void set_path_eigp( std::string path_eigp );

  void save();
  void loadAndCompare();
  void compareMembers( TestResult &ref );
  void reportResult( std::size_t tests, std::size_t fails );
  template<class Archive>
    void serialize(Archive & a, const std::size_t version)
  {
    a & BOOST_SERIALIZATION_NVP( iterationResults );
  }
};


#include "testresult_impl.hpp"

#endif // CHASE_TESTRESULT_H
