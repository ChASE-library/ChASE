/*! \file chfsi.cpp
 *      \brief Core cpp file
 *      \details This file contains some auxiliary functions. It is at top of the '#include tree'.
 *  */ 

/// \cond

#include "../include/chase.hpp"

/// \endcond

static bool elechfsi_stat = true;
static int  elechfsi_filtered;
static int  elechfsi_iteration;
static double elechfsi_time[12] = {0.0};
static int  elechfsi_degmax = 40;
static int  elechfsi_degrees_len;
static int* elechfsi_degrees_ell = NULL;
static int  elechfsi_delta = 3;
static int  elechfsi_maxiter = 15;
static int  elechfsi_lanczos = 10;
 
void swap_perm(int k, int j, int* pi, int* pi_inv)
{
  int tmp = pi[k];
  pi[k] = pi[j];
  pi[j] = tmp;
  
  tmp = pi_inv[pi[k]];
  pi_inv[pi[k]] = pi_inv[pi[j]];
  pi_inv[pi[j]] = tmp;
  return;
}

bool get_stat(){return elechfsi_stat;}
void set_stat(bool stat){elechfsi_stat = stat;}

int  get_filtered(){return elechfsi_filtered;}
void set_filtered(int f){elechfsi_filtered = f;}

int  get_iteration(){return elechfsi_iteration;}
void set_iteration(int i){elechfsi_iteration = i;}


void   get_times(double* time){for(int i = 0; i < 6; ++i) time[i] = elechfsi_time[5+i];}
double get_time(int index){return elechfsi_time[index];}
void   set_time(const int index, const double t){elechfsi_time[index] = t;}

int  get_degmax(void){return elechfsi_degmax;}
void set_degmax(const int degmax){elechfsi_degmax = degmax;}

/** \fn get_degrees(int* degrees, int *deglen)
 * \brief Returns an array of integers, specifying the degrees used for each vector by the filter. It also returns the length of the array.
 * \param degrees               Array of integers, it's elements are the degrees that are used by the filter for each vector. 
 * \param deglen                Integer specifying the length of the array.
 * \return void
 */
void get_degrees(int* degrees, int *deglen)
{
  assert(deglen != NULL);
  *deglen = elechfsi_degrees_len;
  if (degrees == NULL) return;
  for (int i = 0; i < *deglen; ++i)
    degrees[i] = elechfsi_degrees_ell[i];
  return;
}
/** \fn init_degrees(int deglen)
 * \brief Initializes all of the filter degrees with the same value.
 * \param k             Integer value, which is set to all of the elements in the degrees array.
 * \return void
 */
void init_degrees(int deglen)
{
  assert(deglen > 0);
  elechfsi_degrees_len = deglen;
  elechfsi_degrees_ell = new int[deglen]; 
}
void set_degree(int index, int d){elechfsi_degrees_ell[index] = d;}
int  get_degree(int index){return elechfsi_degrees_ell[index];}

int  get_delta(){return elechfsi_delta;}
void set_delta(int d){elechfsi_delta = d;}

int  get_maxiter(){return elechfsi_maxiter;}
void set_maxiter(int m){elechfsi_maxiter = m;}

int  get_lanczos(){return elechfsi_lanczos;}
void set_lanczos(int l){elechfsi_lanczos = l;}
