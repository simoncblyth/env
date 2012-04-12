#
#  C++ runtime dynamic loading , aka plugins 
#     http://www.linuxjournal.com/article/3687
#

CC = g++
CCOPT = -g
SOFLAGS = -dynamiclib -Wl,-dylib -Wl,-undefined -Wl,dynamic_lookup 
NAME1 = circle
NAME2 = square

all:exe libs test
	@echo done


exe:main
	@echo built $(shell ls -l $^)

main: main.o
	$(CC) -rdynamic -ldl $^ -o $@

main.o: main.cc
	$(CC) -I/usr/include $(CCOPT) -c $^ -o $@ 




libs:lib$(NAME1).so lib$(NAME2).so
	@echo built $(shell ls -l $^) 
	@echo $(shell file $^)

###

lib$(NAME1).so:$(NAME1).o
	#ld -dylib -undefined dynamic_lookup -o $@ $^
	$(CC) $(SOFLAGS) $^ -o $@  


$(NAME1).o: $(NAME1).cc
	$(CC) -I/usr/include $(CCOPT) -c $^ -o $@ 

###

lib$(NAME2).so:$(NAME2).o
	#ld -dylib -undefined dynamic_lookup -o $@ $^
	$(CC) $(SOFLAGS) $^ -o $@  

$(NAME2).o: $(NAME2).cc
	$(CC) -I/usr/include $(CCOPT) -c $^ -o $@ 


test:
	./main lib$(NAME1).so $(NAME1)
	./main lib$(NAME2).so $(NAME2)

clean:
	rm -f *.{so,dylib,o} main


.PHONY: exe libs all clean test


