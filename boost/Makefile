NAME = speed

all: $(NAME)
	@echo done


$(NAME) : $(NAME).o
	g++ -L/opt/local/lib  $< -o $@

$(NAME).o : $(NAME).cpp
	g++ -c -I/opt/local/include $< -o $@



.PHONY: all clean


clean : 
	rm *.o speed


