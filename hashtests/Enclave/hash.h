#ifndef HASH_H
#define HASH_H
#include <functional>

size_t float_hash(const float * data, size_t num_floats){
  static std::hash<float> h;
  size_t ret = 0;
  for(size_t i = 0; i < num_floats; i++){
    ret ^= (h(data[i]) << 1);
  }
  return ret;
}

#endif
