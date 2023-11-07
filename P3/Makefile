SHELL = /bin/bash

CC = mpicc
INCLUDE_DIRECTORIES = include
SRC_DIRECTORIES = src
CFLAGS = -std=gnu17 -O3 -march=native -ffast-math -D_GNU_SOURCE -fopenmp -MMD $(addprefix -I ,$(INCLUDE_DIRECTORIES))
LDFLAGS = -lm
OBJECTS = mtx.o main.o
TARGET = main
DEPS = $(OBJECTS:.o=.d)
vpath %.c $(SRC_DIRECTORIES)
vpath %.h $(INCLUDE_DIRECTORIES)

$(TARGET) : $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS) $(LDFLAGS)

-include $(DEPS)

.PHONY : clean
clean :
	rm -f $(TARGET) $(DEPS) $(OBJECTS)