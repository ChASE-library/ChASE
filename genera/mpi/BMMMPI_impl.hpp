/* -*- Mode: C++; tab-width: 8; indent-tabs-mode: nil; c-basic-offset: 2 -*- */
/* vim: set ts=8 sts=2 et sw=2 tw=80: */
/* TODO License */

//This function initializes all arrays, communicators and calculates
//offsets, sizes and dimensions.
template <typename T>
void MPI_handler_init(MPI_Handler<T>* MPI_hand, CHASE_INT global_n, CHASE_INT nev)
{
    //global_n is dimension of problem matrix H
    CHASE_MPIINT periodic[] = { 0, 0 };
    CHASE_MPIINT reorder = 0;
    CHASE_MPIINT coord[2];
    CHASE_MPIINT r;
    //global_n = 22360;
    MPI_hand->global_n = global_n;
    MPI_Comm_size(MPI_COMM_WORLD, &(MPI_hand->nprocs));

    //There must never be more then global_n processes.
    if (MPI_hand->nprocs > MPI_hand->global_n) {
        printf("ERROR: number of processes preceeds dimension of the problem\n");
        return;
    }
    MPI_Comm_rank(MPI_COMM_WORLD, &(MPI_hand->rank));
    MPI_hand->dims[0] = MPI_hand->dims[1] = 0;

    //This creates optimal grid for block decomposition
    MPI_Dims_create(MPI_hand->nprocs, 2, MPI_hand->dims);

    //_____________________________Only for timing purposes_________________//
    //MPI_hand->dims[0] = 16;                                                  //
    //MPI_hand->dims[1] = 1;                                                  //
    //______________________________________________________________________//

    //Useful cartesian communicator
    MPI_Cart_create(MPI_COMM_WORLD, 2, MPI_hand->dims, periodic, reorder, &(MPI_hand->CART_COMM));

    //Calculating ranks which will consist wog and col group and later row and col communicators
    MPI_Cart_coords(MPI_hand->CART_COMM, MPI_hand->rank, 2, MPI_hand->coord);
    MPI_hand->ranks_row = (CHASE_MPIINT*)malloc(MPI_hand->dims[1] * sizeof(CHASE_MPIINT));
    MPI_hand->ranks_col = (CHASE_MPIINT*)malloc(MPI_hand->dims[0] * sizeof(CHASE_MPIINT));
    for (auto i = 0; i < MPI_hand->dims[0]; i++) {
        coord[0] = i;
        coord[1] = MPI_hand->coord[1];
        MPI_Cart_rank(MPI_hand->CART_COMM, coord, &r);
        MPI_hand->ranks_col[i] = r;
    }
    for (auto j = 0; j < MPI_hand->dims[1]; j++) {
        coord[1] = j;
        coord[0] = MPI_hand->coord[0];
        MPI_Cart_rank(MPI_hand->CART_COMM, coord, &r);
        MPI_hand->ranks_row[j] = r;
        MPI_Cart_rank(MPI_hand->CART_COMM, coord, &r);
        MPI_hand->ranks_row[j] = r;
    }
    MPI_Comm_group(MPI_hand->CART_COMM, &(MPI_hand->origGroup));
    MPI_Group_incl(MPI_hand->origGroup, MPI_hand->dims[1], MPI_hand->ranks_row, &(MPI_hand->ROW));
    MPI_Group_incl(MPI_hand->origGroup, MPI_hand->dims[0], MPI_hand->ranks_col, &(MPI_hand->COL));
    MPI_Comm_create(MPI_COMM_WORLD, MPI_hand->ROW, &(MPI_hand->ROW_COMM));
    MPI_Comm_create(MPI_COMM_WORLD, MPI_hand->COL, &(MPI_hand->COL_COMM));

    MPI_hand->nev = nev;
    //offsets of matrix blocks in respect to the full matrix H
    MPI_hand->off[0] = MPI_hand->coord[0] * (MPI_hand->global_n / MPI_hand->dims[0]);
    MPI_hand->off[1] = MPI_hand->coord[1] * (MPI_hand->global_n / MPI_hand->dims[1]);

    //Calculating sizes of arrays saved on this process
    if (MPI_hand->coord[0] < MPI_hand->dims[0] - 1) {
        MPI_hand->m = MPI_hand->global_n / MPI_hand->dims[0];
    } else {
        MPI_hand->m = MPI_hand->global_n - (MPI_hand->dims[0] - 1) * (MPI_hand->global_n / MPI_hand->dims[0]);
    }
    if (MPI_hand->coord[1] < MPI_hand->dims[1] - 1) {
        MPI_hand->n = MPI_hand->global_n / MPI_hand->dims[1];
    } else {
        MPI_hand->n = MPI_hand->global_n - (MPI_hand->dims[1] - 1) * (MPI_hand->global_n / MPI_hand->dims[1]);
    }

#ifdef GPU_MODE
    //Allocation of memory
    cudaMallocHost(&(MPI_hand->A), MPI_hand->m * MPI_hand->n * sizeof(T));
    cudaMallocHost(&(MPI_hand->B), MPI_hand->n * MPI_hand->nev * sizeof(T));
    cudaMallocHost(&(MPI_hand->C), MPI_hand->m * MPI_hand->nev * sizeof(T));
    cudaMallocHost(&(MPI_hand->IMT), (MPI_hand->m > MPI_hand->n ? MPI_hand->m : MPI_hand->n) * MPI_hand->nev * sizeof(T));
#else
    MPI_hand->A = new T[MPI_hand->m * MPI_hand->n]();
    MPI_hand->B = new T[MPI_hand->n * MPI_hand->nev]();
    MPI_hand->C = new T[MPI_hand->m * MPI_hand->nev]();
    MPI_hand->IMT = new T[(MPI_hand->m > MPI_hand->n ? MPI_hand->m : MPI_hand->n) * MPI_hand->nev]();

/*
      // Note: this is untested
    std::mt19937 gen(2342.0); // TODO
    std::normal_distribution<> d;
    for (auto j = 0; j < MPI_hand->n * MPI_hand->nev; j++) {
        if (std::is_same<T, Base<T> >::value)
            MPI_hand->B[j] = T(d(gen));
        else
            MPI_hand->B[j] = T(d(gen), d(gen));
    }
    for (auto j = 0; j < MPI_hand->m * MPI_hand->nev; j++) {
        if (std::is_same<T, Base<T> >::value)
            MPI_hand->C[j] = T(d(gen));
        else
            MPI_hand->C[j] = T(d(gen), d(gen));
    }
    */
#endif

    printf("\n***********************************************************\n");
    printf("MPI Handler on process %d initialized.\n Dimensions of grid: %d, %d.\n", MPI_hand->rank, MPI_hand->dims[0], MPI_hand->dims[1]);
    printf("Block with coordinates [%d,%d] has dimensions %d, %d\n", MPI_hand->coord[0], MPI_hand->coord[1], MPI_hand->m, MPI_hand->n);
    printf("Block with coordinates [%d,%d] has offsets %d, %d\n", MPI_hand->coord[0], MPI_hand->coord[1], MPI_hand->off[0], MPI_hand->off[1]);

    //Initializing these important stuff
    MPI_hand->next = 'b';
    MPI_hand->initialized = 1;
//initializing GPU_MPI_handler which will on its own recognize the GPUS
#ifdef GPU_MODE
    {
        CHASE_INT count;
        cudaError_t err = cudaGetDeviceCount(&count);
        printf("\n******************************************************************************\nNUMBER OF DEVICES ON PROCESS %d: %d\n ", MPI_hand->rank, count);
        printf("%s\n****************************************************************************\n\n", cudaGetErrorString(err));
    }
    GPU_init(MPI_hand->m, MPI_hand->n, MPI_hand->nev, &(MPI_hand->GPU_MPI_hand));
#endif
}

template <typename T>
void MPI_distribute_H(MPI_Handler<T>* MPI_hand, T* H_Full)
{
    for (size_t i = 0; i < MPI_hand->n; i++) {
        for (size_t j = 0; j < MPI_hand->m; j++) {
            MPI_hand->A[i * MPI_hand->m + j] = H_Full[MPI_hand->off[0] + j + (i + MPI_hand->off[1]) * MPI_hand->global_n];
        }
    }
#ifdef GPU_MODE
    GPU_load(MPI_hand->A, &(MPI_hand->GPU_MPI_hand));
#endif
}

template <typename T>
void MPI_distribute_V(MPI_Handler<T>* MPI_hand, T* V, CHASE_INT nev)
{
    MPI_hand->next = 'c';

    for (auto j = 0; j < nev; j++) {
        std::memcpy(MPI_hand->C + j * MPI_hand->m, V + j * MPI_hand->global_n + MPI_hand->off[0], MPI_hand->m * sizeof(T));
        //  for (auto i = 0; i < MPI_hand->m; i++) {
        //    MPI_hand->C[j * MPI_hand->m + i] = V[j * MPI_hand->global_n + i + MPI_hand->off[0]];
        // }
    }

    // for (auto j = 0; j < nev; j++) {
    //     for (auto i = 0; i < MPI_hand->n; i++) {
    //         MPI_hand->B[j * MPI_hand->n + i] = 0; //V[j * MPI_hand->global_n + i + MPI_hand->off[1]];
    //     }
    // }

    // for (auto j = 0; j < nev; j++) {
    //     for (auto i = 0; i < std::max(MPI_hand->m, MPI_hand->n); i++) {
    //         MPI_hand->IMT[j * std::max(MPI_hand->m, MPI_hand->n) + i] = 0;
    //     }
    // }
}

template <typename T>
void MPI_distribute_W(MPI_Handler<T>* MPI_hand, T* V, CHASE_INT nev)
{
    for (auto j = 0; j < nev; j++) {
0        std::memcpy(MPI_hand->B + j * MPI_hand->n, V + j * MPI_hand->global_n + MPI_hand->off[1], MPI_hand->n * sizeof(T));
    }
}


//This function should fill arrays with data
//For now it just generates random data
template <typename T>
void MPI_load(MPI_Handler<T>* MPI_hand)
{
    // initA(MPI_hand);
    // initB(MPI_hand);

    //Sets C to zero initially (This would be starting iteration)
    CHASE_INT k = MPI_hand->m * MPI_hand->nev;
    T alpha = { 0, 0 };
    CHASE_INT incx = 1;
    //scal(&k,&alpha,MPI_hand->C,&incx);
    t_scal(k, &alpha, MPI_hand->C, 1);
//load data onto GPUS
#ifdef GPU_MODE
//GPU_load(MPI_hand->A,&(MPI_hand->GPU_MPI_hand));
#endif
}

//This function performs multiplication. It checks whether it is time for
//A*B or A*C and accordingly calls appropriate zgemm version and appropriate
//communicator for reduction
template <typename T>
void MPI_doGemm(MPI_Handler<T>* MPI_hand, T alpha, T beta, CHASE_INT offset, CHASE_INT nev)
{
    char TRANSA, TRANSB;
    T b;
    //TODO: free these
    T* One = new T(1.0, 0.0);
    T* Zero = new T(0.0, 0.0);
    double start, end;
    CHASE_INT dim, inc;
    if (MPI_hand->next == 'b') {
//printf("\nReduction over rows on process %d.\n",MPI_hand->rank);
//start = MPI_Wtime();
#ifdef GPU_MODE
        GPU_doGemm(MPI_hand->B + offset, MPI_hand->IMT + offset, nev, &(MPI_hand->GPU_MPI_hand), MPI_hand->next);
#else
        t_gemm(CblasColMajor, CblasNoTrans, CblasNoTrans, MPI_hand->m, nev, MPI_hand->n, One, MPI_hand->A, MPI_hand->m, MPI_hand->B + offset * MPI_hand->n, MPI_hand->n, Zero, MPI_hand->IMT + offset * MPI_hand->m, MPI_hand->m);
#endif

        //end = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, MPI_hand->IMT + offset * MPI_hand->m, MPI_hand->m * nev, getMPI_Type<T>(), MPI_SUM, MPI_hand->ROW_COMM);
        dim = MPI_hand->m * nev;
        inc = 1;
        //scal(&dim,&beta,MPI_hand->C+offset,&inc);
        t_scal(dim, &beta, MPI_hand->C + offset * MPI_hand->m, 1);
        //zaxpy(&dim,&alpha,MPI_hand->IMT+offset,&inc,MPI_hand->C+offset,&inc);
        t_axpy(dim, &alpha, MPI_hand->IMT + offset * MPI_hand->m, 1, MPI_hand->C + offset * MPI_hand->m, 1);

        MPI_hand->next = 'c';
        return;
        //return end-start;
    }
    if (MPI_hand->next == 'c') {
//printf("\nReduction over columns on process %d\n", MPI_hand->rank);
//start = MPI_Wtime();
#ifdef GPU_MODE
        GPU_doGemm(MPI_hand->IMT + offset, MPI_hand->C + offset, nev, &(MPI_hand->GPU_MPI_hand), MPI_hand->next);
#else
        t_gemm(CblasColMajor, CblasConjTrans, CblasNoTrans, MPI_hand->n, nev, MPI_hand->m, One, MPI_hand->A, MPI_hand->m, MPI_hand->C + offset * MPI_hand->m, MPI_hand->m, Zero, MPI_hand->IMT + offset * MPI_hand->n, MPI_hand->n);
#endif
        //end = MPI_Wtime();
        MPI_Allreduce(MPI_IN_PLACE, MPI_hand->IMT + offset * MPI_hand->n, MPI_hand->n * nev, getMPI_Type<T>(), MPI_SUM, MPI_hand->COL_COMM);
        dim = MPI_hand->n * nev;
        inc = 1;
        //scal(&dim,&beta,MPI_hand->B+offset,&inc);
        t_scal(dim, &beta, MPI_hand->B + offset * MPI_hand->n, 1);
        //zaxpy(&dim,&alpha,MPI_hand->IMT+offset,&inc,MPI_hand->B+offset,&inc);
        t_axpy(dim, &alpha, MPI_hand->IMT + offset * MPI_hand->n, 1, MPI_hand->B + offset * MPI_hand->n, 1);

        MPI_hand->next = 'b';
        return;
        //return end-start;
    }
    printf("Something is wrong with selecting multiplication mode!\n");
    //	return 0;
}
//Kills arrays in MPI_handler
template <typename T>
void MPI_destroy(MPI_Handler<T>* MPI_hand)
{
    free(MPI_hand->ranks_row);
    free(MPI_hand->ranks_col);
#ifdef GPU_MODE
    cudaFreeHost(MPI_hand->A);
    cudaFreeHost(MPI_hand->B);
    cudaFreeHost(MPI_hand->C);
    GPU_destroy(&(MPI_hand->GPU_MPI_hand));
#endif
}

//This function returns offsets and sizes. It should be used before Load-
template <typename T>
void MPI_get_off(MPI_Handler<T>* MPI_hand, CHASE_INT* xoff, CHASE_INT* yoff, CHASE_INT* xlen, CHASE_INT* ylen)
{
    if (MPI_hand->initialized != 1) {
        printf("ERROR: Impossible to get offsets because MPI_Handler<T> is not initialized!\n");
        return;
    }
    *xoff = MPI_hand->off[0];
    *yoff = MPI_hand->off[1];
    *xlen = MPI_hand->m;
    *ylen = MPI_hand->n;
}

//When we want to stop with multiplication we call this.
template <typename T>
void MPI_get_C(MPI_Handler<T>* MPI_hand, CHASE_INT* COff, CHASE_INT* CLen, T* C, CHASE_INT nev)
{
    if (MPI_hand->initialized != 1) {
        printf("ERROR: Impossible to get C because MPI_Handler<T> is not initialized!\n");
        return;
    }
    CHASE_INT size;
    CHASE_INT inc = 1;
    if (MPI_hand->next == 'b') {
        *COff = MPI_hand->off[1];
        *CLen = MPI_hand->m;
        size = MPI_hand->m * nev;
        // TODO
        //copy(&size,MPI_hand->C,&inc,C,&inc);
        t_copy(size, MPI_hand->B, inc, C, inc);
    } else if (MPI_hand->next == 'c') {
        *COff = MPI_hand->off[0];
        *CLen = MPI_hand->n;
        size = MPI_hand->n * nev;
        // TODO
        //copy(&size,MPI_hand->C,&inc,C,&inc);
        t_copy(size, MPI_hand->C, inc, C, inc);
    } else
        printf("Something is wrong!\n");
    // for( CHASE_INT i = 0; i < size; ++i )
    //   {
    //   MPI_hand->C[i] = std::complex<double>(0.0, 0.0);
    //   MPI_hand->B[i] = std::complex<double>(0.0, 0.0);
    //   MPI_hand->IMT[i] = std::complex<double>(0.0, 0.0);
    //   }
    return;
}

// TODO shift with offsets!
template <typename T>
void shiftA(MPI_Handler<T>* MPI_hand, T c)
{
    for (size_t i = 0; i < MPI_hand->n; i++) {
        for (size_t j = 0; j < MPI_hand->m; j++) {
            if (MPI_hand->off[0] + j == (i + MPI_hand->off[1])) {
                MPI_hand->A[i * MPI_hand->m + j] += c;
            }
        }
    }

#ifdef GPU_MODE
    shiftA(&(MPI_hand->GPU_MPI_hand), c);
#endif
}

template <typename T>
void debug_H(MPI_Handler<T>* MPI_hand)
{
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (auto i = 0; i < size; ++i) {

        if (rank == i) {
            std::cout << "[" << rank << "] A:\n";
            for (auto j = 0; j < MPI_hand->m; j++) {
                for (auto i = 0; i < MPI_hand->n; i++) {
                    std::cout << MPI_hand->A[i * MPI_hand->m + j] << " ";
                }
                std::cout << "\n";
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

template <typename T>
void debug_B(MPI_Handler<T>* MPI_hand)
{
    int rank, localrank;
    int size;
    MPI_Comm_rank(MPI_hand->ROW_COMM, &localrank);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (auto i = 0; i < size; ++i) {

        if (rank == i) {
            std::cout << "[" << localrank << "] B:\n";
            for (auto i = 0; i < MPI_hand->n; i++) {
                for (auto j = 0; j < MPI_hand->nev; j++) {
                    std::cout << MPI_hand->B[j * MPI_hand->n + i] << " ";
                }
                std::cout << "\n";
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

template <typename T>
void debug_C(MPI_Handler<T>* MPI_hand)
{
    int rank;
    int size;
    MPI_Comm_rank(MPI_hand->COL_COMM, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    for (auto i = 0; i < size; ++i) {

        if (rank == i) {
            std::cout << "[" << rank << "] C:\n";
            for (auto i = 0; i < MPI_hand->m; i++) {
                for (auto j = 0; j < MPI_hand->nev; j++) {
                    std::cout << MPI_hand->C[j * MPI_hand->m + i] << " ";
                }
                std::cout << "\n";
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

template <typename T>
void debug_IMT(MPI_Handler<T>* MPI_hand)
{
    for (auto i = 0; i < MPI_hand->m; i++) {
        for (auto j = 0; j < MPI_hand->nev; j++) {
            std::cout << MPI_hand->IMT[j * MPI_hand->n + i] << " ";
        }
        std::cout << "\n";
    }
}

template <typename T>
void MPI_lock_vectors(MPI_Handler<T>* MPI_hand, CHASE_INT nev)
{
    //std::cout << "locking " <<  nev << " vectors\n";
    CHASE_INT size = nev * MPI_hand->n;
    CHASE_INT inc = 1;
    if (MPI_hand->next == 'c')
        //copy(&size,MPI_hand->C,&inc,MPI_hand->B,&inc);
        t_copy(size, MPI_hand->C, inc, MPI_hand->B, inc);
    else //if(MPI_hand->next == 'b')
        //copy(&size,MPI_hand->B,&inc,MPI_hand->C,&inc);
        t_copy(size, MPI_hand->B, inc, MPI_hand->C, inc);
}

template <typename T>
void cpy_vectors(MPI_Handler<T>* MPI_hand, CHASE_INT new_converged, CHASE_INT locked)
{
    CHASE_INT N = MPI_hand->n;
    if (MPI_hand->next == 'c') {
        memcpy(MPI_hand->B + locked * N, MPI_hand->C + locked * N, N * (new_converged) * sizeof(T));
    } else {
        memcpy(MPI_hand->C + locked * N, MPI_hand->B + locked * N, N * (new_converged) * sizeof(T));
    }
}

template <typename T>
void assemble_C(MPI_Handler<T>* MPI_hand, CHASE_INT nevex, T* targetBuf)
{

    int N = MPI_hand->global_n;
    int dimsIdx;
    int subsize;
    T* buff;
    MPI_Comm comm;
    if (MPI_hand->next == 'b') {
        /*
        dim = MPI_hand->n * nev;
        inc = 1;
        t_scal(dim, &beta, MPI_hand->B + offset, 1);
        t_axpy(dim, &alpha, MPI_hand->IMT + offset, 1, MPI_hand->B + offset, 1);
      */
        subsize = MPI_hand->n;
        buff = MPI_hand->B;
        comm = MPI_hand->ROW_COMM;
        dimsIdx = 1;
    } else {
        subsize = MPI_hand->m;
        buff = MPI_hand->C;
        comm = MPI_hand->COL_COMM;
        dimsIdx = 0;
    }

    int gsize, rank;
    MPI_Comm_size(comm, &gsize);
    MPI_Comm_rank(comm, &rank);
    std::vector<int> recvcounts(gsize);
    std::vector<int> displs(gsize);
    //int* recvcounts = (int*)malloc(gsize * sizeof(int));
    //int* displs = (int*)malloc(gsize * sizeof(int));

    for (auto i = 0; i < gsize; ++i) {
        recvcounts[i] = MPI_hand->global_n / MPI_hand->dims[dimsIdx];
        displs[i] = i * recvcounts[0];
    }
    recvcounts[gsize - 1] = MPI_hand->global_n - (MPI_hand->dims[dimsIdx] - 1) * (MPI_hand->global_n / MPI_hand->dims[dimsIdx]);

    /*
    std::cout << "N: " << N << "\n";
    std::cout << "nev: " << nevex << "\n";
    std::cout << "size: " << gsize << "\n";
    std::cout << "dims: " << MPI_hand->dims[dimsIdx] << "\n";
    std::cout << "next: " << MPI_hand->next << "\n";
    std::cout << "subsize: " << subsize << "\n";


    for (auto i = 0; i < gsize; ++i) {
        std::cout << i << "   count: " << recvcounts[i] << " " << displs[i] << "\n";
    }

*/

    //TODO
    // for (auto i = 0; i < nevex; ++i)
    // {
    //     MPI_Allgatherv(
    //         buff + subsize * i, subsize, getMPI_Type<T>(),
    //         targetBuf + i * N, recvcounts, displs, getMPI_Type<T>(),
    //         comm);
    // }

    std::vector<MPI_Request> reqs(gsize);
    std::vector<MPI_Datatype> newType(gsize);

    // Set up the datatype for the recv
    for (auto i = 0; i < gsize; ++i) {

        int array_of_sizes[2] = { MPI_hand->global_n, nevex };
        int array_of_subsizes[2] = { recvcounts[i], nevex };
        int array_of_starts[2] = { displs[i], 0 };

        MPI_Type_create_subarray(
            2,
            array_of_sizes,
            array_of_subsizes,
            array_of_starts,
            MPI_ORDER_FORTRAN,
            getMPI_Type<T>(),
            &(newType[i]));

        MPI_Type_commit(&(newType[i]));
    }

    for (auto i = 0; i < gsize; ++i) {
        if (rank == i) {
            // The sender sends from the appropriate buffer
            MPI_Ibcast(buff, recvcounts[i] * nevex, getMPI_Type<T>(), i, comm, &reqs[i]);
        } else {
            //MPI_Bcast(MPI_hand->C, recvcounts[i] * nevex, getMPI_Type<T>(), i, comm);
            // The recv goes right unto the correct bugger
            MPI_Ibcast(targetBuf, 1, newType[i], i, comm, &reqs[i]);
        }
    }

    // we copy the sender into the target Buffer directly
    int i = rank;
    for (auto j = 0; j < nevex; ++j) {
        std::memcpy(targetBuf + j * N + displs[i], buff + recvcounts[i] * j, recvcounts[i] * sizeof(T));
        // for (auto k = 0; k < recvcounts[i]; ++k) {
        //     targetBuf[j * N + k + displs[i]] = buff[k + recvcounts[i] * j];
        // }
    }

    MPI_Waitall(gsize, reqs.data(), MPI_STATUSES_IGNORE);

    for (auto i = 0; i < gsize; ++i) {
        MPI_Type_free(&newType[i]);
    }
}
