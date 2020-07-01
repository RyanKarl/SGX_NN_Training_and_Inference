//Enclave.h
//Jonathan S. Takeshita, Ryan Karl, Mark Horeni

#ifndef ENCLAVE_H
#define ENCLAVE_H

#if defined(__cplusplus)
extern "C" {
#endif

void start_timing();

void finish_timing();

void print_timings(std::ostream & os);

void reserve_vec(unsigned int n);

int enclave_main(unsigned int n, unsigned int s);

#if defined(__cplusplus)
}
#endif

#endif
