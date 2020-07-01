#ifndef OCALLS_H
#define OCALLS_H

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <unistd.h>
#include <math.h>
#include <fcntl.h>
#include <string.h>
#include <limits.h>

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>

using std::vector;
using std::string;
using std::ifstream;
using std::ofstream;
using std::cout;
using std::cerr;
using std::endl;
using namespace std::chrono;


static high_resolution_clock::time_point hash_start;

static std::vector<double> hash_times;

void reserve_vec(unsigned int n){
  hash_times.reserve(n);
}

void start_timing(){
  hash_start = high_resolution_clock::now();
  return;
}

void finish_timing(){
  high_resolution_clock::time_point end = high_resolution_clock::now();
  hash_times.push_back(duration_cast<std::chrono::nanoseconds>(end - hash_start).count());
  return;
}

//Not technically an OCALL, but should be in this file to access stored timings
void print_timings(std::ostream & os){
  for(const double & d : hash_times){
    os << d << '\n';
  }
}


#endif
