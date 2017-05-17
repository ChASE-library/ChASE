/* -*- Mode: C; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */
#include <assert.h>
#include <complex.h>
#include <hdf5.h>
#include <mpi.h>
#include <mpi.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#define FLOAT_TYPE float
#define FLOAT_TYPE_HDF H5T_NATIVE_FLOAT

typedef struct complex_t {
    FLOAT_TYPE re; /*real part */
    FLOAT_TYPE im; /*imaginary part */
} complex_t;

#define HDF_FILE "work/slai/slai10/array_%d.hdf5"

void chase_write_hdf5(MPI_Comm comm, FLOAT_TYPE complex* H, size_t N_)
{

    int myrank;
    int nprocs;

    int N = N_;

    //////////////////////////
    // Write Matrix to file //
    //////////////////////////

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    MPI_Info info = MPI_INFO_NULL;
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);

    char filename[100];
    sprintf(filename, HDF_FILE, nprocs);

    hid_t file_id, dset_id;
    file_id = H5Fcreate(filename, H5F_ACC_TRUNC, H5P_DEFAULT, plist_id);
    H5Pclose(plist_id);

    hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
    H5Tinsert(complex_id, "real", HOFFSET(complex_t, re), FLOAT_TYPE_HDF);
    H5Tinsert(complex_id, "imag", HOFFSET(complex_t, im), FLOAT_TYPE_HDF);

    hsize_t dimsf[2]; // dataset dimensions
    dimsf[0] = (hsize_t)N;
    dimsf[1] = (hsize_t)N;

    // filespace
    hid_t fspace = H5Screate_simple(2, dimsf, NULL);
    dset_id = H5Dcreate(file_id, "Hamiltonian", complex_id, fspace, H5P_DEFAULT,
        H5P_DEFAULT, H5P_DEFAULT);
    H5Sclose(fspace);

    size_t* matdim_node = (size_t*)malloc(sizeof(size_t) * nprocs);
    size_t* offsets_node = (size_t*)malloc(sizeof(size_t) * (nprocs + 1));
    hsize_t offset[2]; // Start of hyperslab
    hsize_t count[2]; // Stride of hyperslab
    offsets_node[0] = 0;
    for (size_t i = 0; i < nprocs; ++i) {
        matdim_node[i] = N / nprocs;
        if (i < (N % nprocs))
            matdim_node[i] += 1;
        offsets_node[i + 1] = offsets_node[i] + matdim_node[i];
    }

    count[0] = N;
    count[1] = matdim_node[myrank];

    // memspace
    hid_t mspace1 = H5Screate_simple(2, count, NULL);

    offset[0] = 0;
    offset[1] = offsets_node[myrank];

    fspace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    H5Dwrite(dset_id, complex_id, mspace1, fspace, plist_id, H);

    H5Dclose(dset_id);
    H5Sclose(fspace);
    H5Sclose(mspace1);
    H5Pclose(plist_id);
    H5Fclose(file_id);

    // free(data);
    free(matdim_node);
    free(offsets_node);
}

void chase_read_matrix(MPI_Comm comm, size_t xoff, size_t yoff, size_t xlen,
    size_t ylen, FLOAT_TYPE complex* H)
{
    int myrank;
    int nprocs;

    MPI_Comm_size(comm, &nprocs);
    MPI_Comm_rank(comm, &myrank);

    char filename[100];
    sprintf(filename, HDF_FILE, nprocs);

    MPI_Info info = MPI_INFO_NULL;
    hid_t plist_id = H5Pcreate(H5P_FILE_ACCESS);
    H5Pset_fapl_mpio(plist_id, comm, info);

    hid_t file_id, dset_id;
    // H5File *file = new H5File( "array.hdf5", H5F_ACC_TRUNC );
    file_id = H5Fopen(filename, H5F_ACC_RDONLY, plist_id);
    H5Pclose(plist_id);

    dset_id = H5Dopen(file_id, "Hamiltonian", H5P_DEFAULT);
    //hid_t fspace = H5Dget_space(dset_id);

    plist_id = H5Pcreate(H5P_DATASET_XFER);
    H5Pset_dxpl_mpio(plist_id, H5FD_MPIO_COLLECTIVE);

    hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
    H5Tinsert(complex_id, "real", HOFFSET(complex_t, re), FLOAT_TYPE_HDF);
    H5Tinsert(complex_id, "imag", HOFFSET(complex_t, im), FLOAT_TYPE_HDF);

    hsize_t count[2];
    hsize_t offset[2];
    offset[1] = xoff;
    offset[0] = yoff;
    count[1] = xlen;
    count[0] = ylen;

    //memspace
    hid_t mspace1 = H5Screate_simple(2, count, NULL);
    hid_t fspace = H5Dget_space(dset_id);
    H5Sselect_hyperslab(fspace, H5S_SELECT_SET, offset, NULL, count, NULL);

    H5Dread(dset_id, complex_id, mspace1, fspace, plist_id, H);

    H5Dclose(dset_id);
    H5Sclose(fspace);
    H5Sclose(mspace1);
    H5Pclose(plist_id);
    H5Fclose(file_id);

    //    free(dims);
}
