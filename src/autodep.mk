
# This defines build rules for c/c++/cuda sources with automatic dependency generation

SRCS ?= *.c *.cpp *.cc *.cxx *.cu
#SRCS ?= $(wildcard *.cpp) $(wildcard *.cc) $(wildcard *.cxx) $(wildcard *.cu)

DEPDIR := .deps
$(shell mkdir -p $(DEPDIR) >/dev/null)
DEPFLAGS = -MT $@ -MMD -MP -MF $(DEPDIR)/$*.Td

GCCFLAGS += -fmessage-length=80 #-fdiagnostics-color=auto

DEPBUILD.c   = $(CC)  -x c   $(DEPFLAGS) $(CFLAGS)    $(CPPFLAGS) -E
DEPBUILD.cc  = $(CXX) -x c++ $(DEPFLAGS) $(CXXFLAGS)  $(CPPFLAGS) -E
COMPILE.c    = $(CC)   $(CFLAGS)    $(CPPFLAGS) $(GCCFLAGS) $(TARGET_ARCH) -c
COMPILE.cc   = $(CXX)  $(CXXFLAGS)  $(CPPFLAGS) $(GCCFLAGS) $(TARGET_ARCH) -c
COMPILE.nvcc = $(NVCC) $(NVCCFLAGS) $(CPPFLAGS) -Xcompiler "$(GCCFLAGS)" $(TARGET_ARCH) -c
POSTCOMPILE = mv -f $(DEPDIR)/$*.Td $(DEPDIR)/$*.d

CLEAR_LINE = tput el && echo -n "\033[2K" || true # Clears the current line

%.o : %.c
%.o : %.c $(DEPDIR)/%.d
	@$(CLEAR_LINE)
	@echo -n "Building C source file $<\r"
	@$(DEPBUILD.c) $< > /dev/null
	$(COMPILE.c) $(OUTPUT_OPTION) $<
	@$(POSTCOMPILE)

%.o : %.cpp
%.o : %.cpp $(DEPDIR)/%.d
	@$(CLEAR_LINE)
	@echo -n "Building C++ source file $<\r"
	$(DEPBUILD.cc) $< > /dev/null
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	$(POSTCOMPILE)

%.o : %.cc
%.o : %.cc $(DEPDIR)/%.d
	@$(CLEAR_LINE)
	@echo -n "Building C++ source file $<\r"
	@$(DEPBUILD.cc) $< > /dev/null
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	@$(POSTCOMPILE)

%.o : %.cxx
%.o : %.cxx $(DEPDIR)/%.d
	@$(CLEAR_LINE)
	@echo -n "Building C++ source file $<\r"
	@$(DEPBUILD.cc) $< > /dev/null
	$(COMPILE.cc) $(OUTPUT_OPTION) $<
	@$(POSTCOMPILE)

%.o : %.cu
%.o : %.cu $(DEPDIR)/%.d
	@$(CLEAR_LINE)
	@echo -n "Building CUDA source file $<\r"
	@$(DEPBUILD.cc) $< > /dev/null
	$(COMPILE.nvcc) $(OUTPUT_OPTION) $<
	@$(POSTCOMPILE)

$(DEPDIR)/%.d: ;
.PRECIOUS: $(DEPDIR)/%.d

-include $(patsubst %,$(DEPDIR)/%.d,$(basename $(SRCS)))
