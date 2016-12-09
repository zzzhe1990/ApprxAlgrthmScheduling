CXX			=	nvcc

RM			=	rm -f

NAME		=	paroutT16F301

CPPFLAGS	=	 -g

SRC			=	DPCuda.cu \
				Parallel-PTAS-4Oct-2016.cu

all: $(NAME)

$(NAME):
	$(CXX) $(SRC) -o $(NAME) $(CPPFLAGS)

clean:
	$(RM) $(NAME)

fclean: clean
	$(RM) *~

re:	fclean all
