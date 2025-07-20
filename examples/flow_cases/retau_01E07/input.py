#!/usr/bin/python
# values should be consistent with dns.in
h     = 1.
ub    = 1.
visci = 4.28E+08
#
uconv = 0. # if we solve on a convective reference frame; else = 0.
#
# parameters for averaging
#
tbeg   = 16000.
tend   = 25600.
fldstp = 100
#
# case name (e.g., the Retau)
#
casename = '01E07'.zfill(5)
