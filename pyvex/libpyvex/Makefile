TARGET_EXEC ?= vex.so

BUILD_DIR ?= .
SRC_DIRS ?= .

SRCS := $(shell find $(SRC_DIRS) -name '*.cpp' -or -name '*.c')
OBJS := $(SRCS:%=$(BUILD_DIR)/%.o)
DEPS := $(OBJS:.o=.d)

INC_DIRS := $(shell find $(SRC_DIRS) -type d)
INC_FLAGS := $(addprefix -I,$(INC_DIRS))

CPPFLAGS ?= $(INC_FLAGS) -MMD -MP -lstdc++ -fPIC

$(BUILD_DIR)/$(TARGET_EXEC): $(OBJS)
	$(CXX) -shared $(OBJS) -o $@ $(LDFLAGS)
	cp $(BUILD_DIR)/$(TARGET_EXEC) ..

# c source
$(BUILD_DIR)/%.c.o: %.c
	$(MKDIR_P) $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

# c++ source
$(BUILD_DIR)/%.cpp.o: %.cpp
	$(MKDIR_P) $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@


.PHONY: clean

clean:
	$(RM) *.o
	$(RM) *.d

-include $(DEPS)

MKDIR_P ?= mkdir -p
