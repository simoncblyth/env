! https://github.com/numpy/numpy/issues/4519
! f2py -m Fortran -c chararraytest.f90

module CharArrayTest
  implicit none
contains

subroutine charArrayIn(myStrList)
  character(len=*), intent(in) :: myStrList(:)
  integer :: i

  write(*,*) 'Called "charArrayIn" with args:'
  do i = 1, size(myStrList), 1
    write(*,*) i, trim(myStrList(i))
  end do

end subroutine charArrayIn
end module CharArrayTest

