#ifndef NGAP_KERNELS_H_
#define NGAP_KERNELS_H_

#include "graph.h"
#include "group_graph.h"
#include "my_bitset.h"
#include "ngap_buffer.h"

// O0
__global__ void advanceAndFilterBlockingGroups(
    BlockingBuffer blb, uint8_t *arr_input_streams, int arr_input_streams_size,
    GroupMatchset gms, GroupNodeAttrs gna, GroupAAS gaas, GroupCsr gcsr);

// NAP
template <bool unique, bool record_fs>
__global__ void advanceAndFilterNonBlockingNAPGroups(
    NonBlockingBuffer nblb, uint8_t *arr_input_streams,
    int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,
    GroupAAS gaas, GroupCsr gcsr);

// O1
template <bool unique, bool record_fs>
__global__ void advanceAndFilterNonBlockingGroups(NonBlockingBuffer nblb,
                                                  uint8_t *arr_input_streams,
                                                  int arr_input_streams_size,
                                                  GroupMatchset gms,
                                                  GroupNodeAttrs gna,
                                                  GroupAAS gaas, GroupCsr gcsr);

// O3
template <bool unique, int precompute_depth, bool record_fs>
__global__ void advanceAndFilterNonBlockingPrecGroups(
    NonBlockingBuffer nblb, uint8_t *arr_input_streams,
    int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,
    GroupAAS gaas, GroupCsr gcsr);

// O4
template <bool unique>
__global__ void advanceAndFilterNonBlockingR1Groups(
    NonBlockingBuffer nblb, uint8_t *arr_input_streams,
    int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,
    GroupAAS gaas, GroupCsr gcsr);

template <bool unique>
__global__ void advanceAndFilterNonBlockingR2Groups(
    NonBlockingBuffer nblb, uint8_t *arr_input_streams,
    int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,
    GroupAAS gaas, GroupCsr gcsr);

// OA
template <bool unique, int precompute_depth, bool record_fs, bool adaptive_aas>
__global__ void advanceAndFilterNonBlockingAllGroups(
    NonBlockingBuffer nblb, uint8_t *arr_input_streams,
    int arr_input_streams_size, GroupMatchset gms, GroupNodeAttrs gna,
    GroupAAS gaas, GroupCsr gcsr);

#endif