; http://cs.lmu.edu/~ray/notes/nasmtutorial/
; ----------------------------------------------------------------------------------------
; Writes "Hola, mundo" to the console using a C library. Runs on Linux or any other system
; that does not use underscores for symbols in its C library. To assemble and run:
;
;     nasm -felf64 hola.asm && gcc hola.o && ./a.out
; ----------------------------------------------------------------------------------------

        global  main
        extern  puts

        section .text
main:                                   ; This is called by the C library startup code
        mov     rdi, message            ; First integer (or pointer) argument in rdi
        call    puts                    ; puts(message)
        ret                             ; Return from main back into C library wrapper
message:
        db      "Hola, mundo", 0        ; Note strings must be terminated with 0 in C
