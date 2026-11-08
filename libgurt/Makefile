# Plain-make build (no CMake needed):  make   ->  libgurt.so / libgurt.dylib / gurt.dll
CC      ?= cc
CFLAGS  ?= -O2 -std=c99 -Wall -Wextra -fPIC -fvisibility=hidden
SRC      = src/despeckle.c src/deglitch.c src/gapfill.c src/unfold.c src/tormodel.c $(wildcard src/axisym.c)   # axisym.c exists in the full tree only
UNAME   := $(shell uname -s)
ifeq ($(UNAME),Darwin)
  LIB = libgurt.dylib
  LDFLAGS += -dynamiclib
else ifneq (,$(findstring MINGW,$(UNAME))$(findstring MSYS,$(UNAME)))
  LIB = gurt.dll
  LDFLAGS += -shared
else
  LIB = libgurt.so
  LDFLAGS += -shared
endif
all: $(LIB)
$(LIB): $(SRC) include/gurt.h src/util.h $(wildcard include/gurt_axisym.h)
	$(CC) $(CFLAGS) -Iinclude $(SRC) -o $@ $(LDFLAGS) -lm
clean:
	rm -f libgurt.so libgurt.dylib gurt.dll
.PHONY: all clean
