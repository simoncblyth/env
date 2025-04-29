!!  see search.sh
!! 
!
!! changed index to idx to avoid compilation warning
!!

subroutine find_first(needle, haystack, haystack_length, idx)
    !!
    !! Find the first index of `needle` in `haystack`.
    !!
    implicit none
    integer, intent(in) :: needle
    integer, intent(in) :: haystack_length
    integer, intent(in), dimension(haystack_length) :: haystack
!f2py intent(inplace) haystack
    integer, intent(out) :: idx
    integer :: k

    idx = -1
    do k = 1, haystack_length
        if (haystack(k)==needle) then
            idx = k - 1
            exit
        endif
    enddo
end
