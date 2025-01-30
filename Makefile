CC=gcc
CFLAGS=-g -Wall -march=native -flto -fPIC -I. -Iquantizing
LDFLAGS=-lm
QUANTIZE_DIR=quantizing

all: train test predict summary libnn.so quantize

libnn.so: nn.o data_prep.o
	$(CC) -shared -Wl,-soname,libnn.so -o $@ $^ $(LDFLAGS)

data_prep.o: data_prep.c data_prep.h
	$(CC) $(CFLAGS) -c -o $@ $<

nn.o: nn.c nn.h
	$(CC) $(CFLAGS) -c -o $@ $<

train: train.c nn.o data_prep.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

test: test.c nn.o data_prep.o $(QUANTIZE_DIR)/quantize.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

predict: predict.c nn.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

summary: summary.c nn.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

quantize: $(QUANTIZE_DIR)/quantize_main.o $(QUANTIZE_DIR)/quantize.o nn.o
	$(CC) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(QUANTIZE_DIR)/quantize.o: $(QUANTIZE_DIR)/quantize.c $(QUANTIZE_DIR)/quantize.h nn.h
	$(CC) $(CFLAGS) -c -o $@ $<

$(QUANTIZE_DIR)/quantize_main.o: $(QUANTIZE_DIR)/quantize_main.c $(QUANTIZE_DIR)/quantize.h
	$(CC) $(CFLAGS) -c -o $@ $<

clean:
	rm -f *.o $(QUANTIZE_DIR)/*.o libnn.so train test predict summary quantize model.txt quantized_model.txt

.PHONY: all clean