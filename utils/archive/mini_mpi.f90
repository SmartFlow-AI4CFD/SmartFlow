! filepath: /home/mchao/code/smartsim-practice/mini.f90
! gfortran mini.f90 -o mini -I/leonardo/home/userexternal/mxiao000/code/smartredis/install/include -L/leonardo/home/userexternal/mxiao000/code/smartredis/install/lib -lsmartredis -lsmartredis-fortran
program main
    use smartredis_client, only: CLIENT_TYPE
    use mpi
    implicit none
    type(CLIENT_TYPE) :: client
    integer :: error, myid, ierr
    logical :: is_error, db_clustered
    character(len=100) :: key
    real, dimension(10) :: state_global

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, myid, ierr)

    db_clustered = .false.
    key = 'key2'
    state_global = (/ 2, 2, 3, 4, 5, 6, 7, 8, 9, 10 /)

    print *, 'key2 started'

    call init_smartredis(db_clustered, client, myid)

    if(myid == 0) then
        error = client%put_tensor(trim(key), state_global, shape(state_global))
        is_error = client%SR_error_parser(error)
        if (is_error) then
            stop 'Error in SmartRedis put_tensor operation.'
        end if
    end if

    call MPI_Finalize(ierr)
    print *, 'key 2 finished'
end program main

subroutine init_smartredis(db_clustered, client, myid)
    use smartredis_client, only: CLIENT_TYPE
    implicit none
    logical, intent(in) :: db_clustered
    integer, intent(in) :: myid
    type(CLIENT_TYPE), intent(inout) :: client
    integer :: error
    logical :: is_error
    if (myid == 0) then
      error = client%initialize(db_clustered)
      is_error = client%SR_error_parser(error)
      if (is_error) stop 'Error in SmartRedis client initialization.'
    end if
end subroutine init_smartredis