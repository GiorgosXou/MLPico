CC = gcc
CFLAGS = -std=c89 -D SUPPRESS_PRAGMA_MESSAGE
LDFLAGS = -lm

SOURCES := $(wildcard *.c)
TARGETS := $(SOURCES:.c=)

all: $(TARGETS)

%: %.c
	$(CC) $< -o $@ $(CFLAGS) $(LDFLAGS)

clean:
	rm -f $(TARGETS)
