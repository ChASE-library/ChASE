#ifndef CHASE_TESTRESULT
#define CHASE_TESTRESULT

#include <unordered_map>
#include <fstream>
#include <vector>

#include <boost/archive/xml_oarchive.hpp>
#include <boost/archive/xml_iarchive.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/vector.hpp>

#define CHASE_TESTRESULT_WRITE true
#define CHASE_TESTRESULT_COMPARE false

class TestResultIteration {
public:
  friend class boost::serialization::access;
  friend std::ostream & operator<<(std::ostream &os, const TestResultIteration &tr);

  TestResultIteration();
  TestResultIteration( bool compare_ );

  void compareMembers( TestResultIteration &ref, std::size_t &tests, std::size_t &fails );
  void registerValue( std::string key, int value );
  void registerValue( std::string key, double value );

private:
  const bool compare;
  std::unordered_map< std::string, int > intMap;
  std::unordered_map< std::string, double > doubleMap;

  template<class Archive>
    void serialize(Archive & ar, const unsigned int version)
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
              int n, int nev, int nex, int deg,
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

  const bool compare;
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
    void serialize(Archive & a, const unsigned int version)
  {
    a & BOOST_SERIALIZATION_NVP( iterationResults );
  }
};

#endif // CHASE_TESTRESULT
