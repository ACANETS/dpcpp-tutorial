# `word-count` Sample

Word Count example searches through a text file and count the occurences of given keywords. On Linux, it is similar to "grep", which can output every occurence of a keyword and then count with "wc" tool.  

| Optimized for                     | Description
|:---                               |:---
| OS                                | Linux* Ubuntu* 18.04
| Hardware                          | Skylake with GEN9 or newer, Intel(R) Programmable Acceleration Card with Intel(R) Stratix(R) 10 GX FPGA
| Software                          | Intel&reg; oneAPI DPC++ Compiler (beta)  

## Purpose

We use word-count to show how to design MapReduce programs with DPC++, which can target CPU, GPU and FPGAs. See [README.md](./README.md) for design details.

The following instructions are specifically for the hands-on session of oneAPI DevSummit at ISC'21.

## Steps 

1. Log in DevCloud
```
    ssh devcloud
```

2. Select a compute node
```
    devcloud_login
```
    choose '5' for '5) Compilation (Command Line) Only'
    or 
    choose '2' for '2) Arria 10 - OneAPI, OpenVINO'

3. Set up development environment
```
    tools_setup
```
    choose '6' for '6) Arria 10 - OneAPI, OpenVINO'

4. Download source code
```
    git clone https://github.com/ACANETS/dpcpp-tutorial
```

5. Build 'word-count' for emulation
```
    cd dpcpp-tutorial/word-count
    mkdir build
    cd build
    make
``` 

6. Execute in emulation mode
```
    ./word-count.fpga_emu
```

7. Static Analysis
```
    make report
    cd word-count_report.prj
    tar zcvf reports.tgz reports/
    pwd
    <record this full path to build dir>
```
On your local computer
```
    scp devcloud:<full path to build dir>/reports.tgz .
    tar zxvf reports.tgz
```
Open in your Web browser the report.html file in reports folder.


