subroutine foo(A)
   character, dimension(:,:), intent(in) :: A
   character(len=96), allocatable :: B(:)
   integer :: strLen
   integer :: arrLen

   strLen = size(A,1)
   arrLen = size(A,2)

   allocate(B(arrLen))
   do i = 1, arrLen
      B(i) = transfer(A(:,i),B(i)(1:strLen))
   enddo
   
   do i=1, arrLen
      print*,trim(B(i))
   enddo

end


