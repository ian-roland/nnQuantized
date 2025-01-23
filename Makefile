CC=gcc
CFLAGS=-g -Wall -march=native -flto -fPIC
LDFLAGS=-lm
QUANTIZE_DIR=quantizing
INCLUDES=-I.

all: train test predict summary libnn.so quantize

libnn.so: nn.o data_prep.o
	$(RM) $@
	$(AR) rcs $@ nn.o data_prep.o
	$(CC) -shared -Wl,-o libnn.so *.o

data_prep.o: data_prep.c data_prep.h
	$(CC) -Wall data_prep.c -c -march=native -flto $(CFLAGS) -fPIC

nn.o: nn.c nn.h
	$(CC) -Wall nn.c -c -march=native -flto $(CFLAGS) -fPIC

train: train.c nn.o data_prep.o
	$(CC) -Wall train.c data_prep.o nn.o -o train -lm -march=native $(CFLAGS)

test: test.c nn.o data_prep.o
	$(CC) -Wall test.c data_prep.o nn.o -o test -lm -march=native $(CFLAGS)

predict: predict.c nn.o
	$(CC) -Wall predict.c nn.o -o predict -lm -march=native $(CFLAGS)

summary: summary.c nn.o
	$(CC) -Wall summary.c nn.o -o summary -lm -march=native $(CFLAGS)

quantize: $(QUANTIZE_DIR)/quantize.o nn.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(QUANTIZE_DIR)/quantize.o: $(QUANTIZE_DIR)/quantize.c $(QUANTIZE_DIR)/quantize.h nn.h
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

tags:
	ctags -R *

check:
	cppcheck --enable=all --inconclusive .

clean:
	$(RM) data_prep.o nn.o libnn.so train test predict summary model.txt tags nn.png
	$(RM) -r **pycache**