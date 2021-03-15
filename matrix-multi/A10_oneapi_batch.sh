# Copyright 2020 Intel Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or substantial portions
# of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED
# TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
# CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

###########################################################################################################
# The following flow assumes src directory already exist 
###########################################################################################################

# Initial Setup
hostname
source /data/intel_fpga/devcloudLoginToolSetup.sh
tools_setup -t A10OAPI

# Running project in Emulation mode
printf "\\n%s\\n" "Running in Emulation Mode:"
cd ~/projects/dpcpp-tutorial/matrix-multi
mkdir -p newbuild
cd newbuild
cmake ..
make 
./matrix-multi-para-v1.fpga_emu
error_check

# Running project in FPGA Hardware Mode (this takes approximately 1 hour)
printf "\\n%s\\n" "Building for FPGA profiling:"
cd ~/projects/dpcpp-tutorial/matrix-multi
mkdir -p newbuild
cd newbuild
cmake ..
make profile
error_check

