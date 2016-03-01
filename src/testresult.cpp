#include "../include/testresult.h"
#include "../include/chfsi.h"

#define ANSI_FG_BLACK   "\x1b[30m"
#define ANSI_FG_RED     "\x1b[31m"
#define ANSI_FG_GREEN   "\x1b[32m"
#define ANSI_FG_YELLOW  "\x1b[33m"
#define ANSI_FG_BLUE    "\x1b[34m"
#define ANSI_FG_MAGENTA "\x1b[35m"
#define ANSI_FG_CYAN    "\x1b[36m"
#define ANSI_FG_WHITE   "\x1b[37m"

#define ANSI_BG_RED     "\x1b[41m"
#define ANSI_BG_GREEN   "\x1b[42m"
#define ANSI_BG_YELLOW  "\x1b[43m"
#define ANSI_BG_BLUE    "\x1b[44m"
#define ANSI_BG_MAGENTA "\x1b[45m"
#define ANSI_BG_CYAN    "\x1b[46m"
#define ANSI_BG_WHITE   "\x1b[47m"
#define ANSI_RESET      "\x1b[0m"

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

TestResultIteration::TestResultIteration()
  : compare( CHASE_TESTRESULT_COMPARE )
    {};

TestResultIteration::TestResultIteration( bool compare_ )
  : compare( compare_ )
    {};

void TestResultIteration::compareMembers( TestResultIteration &ref,
                                          std::size_t &tests, std::size_t &fails ) {
    for( auto it = intMap.begin(); it != intMap.end(); ++it )
      assertEqual( it, ref.intMap, tests, fails );
    for( auto it = doubleMap.begin(); it != doubleMap.end(); ++it )
      assertEqual<double>( it, ref.doubleMap, 10-8, tests, fails );
  }

  void TestResultIteration::registerValue( std::string key, int value )
  {
    intMap.insert({ key, value });
  }

  void TestResultIteration::registerValue( std::string key, double value )
  {
    doubleMap.insert({ key, value });
  }

////////////////////////////////////////////////////////////////////////////////

TestResult::TestResult()
  : compare(CHASE_TESTRESULT_COMPARE)
  {};

TestResult::TestResult( bool compare_, std::string name_ )
  : compare(compare_),
    fileName(name_+".xml")
    {};


TestResult::TestResult( bool compare_, std::string name_,
                        int n_, int nev_, int nex_, int deg_,
                        double tol_, char mode_, char opt_, bool sequence)
  : compare(compare_)
    {
      std::ostringstream fileNameBuilder(std::ostringstream::ate);
      fileNameBuilder
        << name_ << "_"
        << n_ << "_"
        <<  nev_ << "_"
        <<  nex_  << "_"
        <<  deg_  << "_"
        <<  tol_ << "_"
        <<  mode_  << "_"
        <<  opt_;

      if( sequence )
        fileNameBuilder << "_seq";
      
      fileNameBuilder << ".xml";
      fileName = fileNameBuilder.str();
    };

  std::string TestResult::name()
  {
    return this->fileName;
  }

  void TestResult::done() {
    // either save or compare
    if( compare == CHASE_TESTRESULT_WRITE )
      save();
    else
    {
      // load reference
      loadAndCompare();
    }
  }
  void TestResult::save() {
    std::ofstream ofs(this->name().c_str());
    boost::archive::xml_oarchive oa(ofs);
    oa << BOOST_SERIALIZATION_NVP(this);
  }

  void TestResult::loadAndCompare() {
    // construct object
    TestResult ref;
    std::ifstream ifs(this->name().c_str());

    try
    {
      boost::archive::xml_iarchive ia(ifs);
      ia >> BOOST_SERIALIZATION_NVP(ref);
    }
    catch( std::exception &e )
    {
      std::cout << "File not found, writing" << std::endl
                << "Consider creating the test profile for the whole sequence" << std::endl;

      save();
      return;
    }

    compareMembers( ref );
  }

  void TestResult::compareMembers( TestResult &ref ) {
    std::size_t tests=0;
    std::size_t fails=0;

    if( iterationResults.size() > ref.iterationResults.size() )
      std::cout << "Reference does not contain full sequence!" << std::endl
                << "Consider creating the test profile for the whole sequence" << std::endl;

    for( auto it = 0;
         it < std::min( iterationResults.size(), ref.iterationResults.size() );
         ++it )
      iterationResults[it].compareMembers( ref.iterationResults[it], tests, fails );
    reportResult( tests, fails );
  }

void TestResult::reportResult( std::size_t tests, std::size_t fails )
{
  if( fails == 0 )
    std::cout  << ANSI_BG_GREEN;
    else
      std::cout <<  ANSI_BG_RED;
  std::cout <<  ANSI_FG_BLACK << "PASSED\t"  << (tests - fails) << " / " << tests << ANSI_RESET << std::endl;
}
