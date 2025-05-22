#!/usr/bin/python
# values should be consistent with dns.in
h     = 1.
ub    = 1.
visci = 5.50E+10
#
uconv = 0. # if we solve on a convective reference frame; else = 0.
#
# parameters for averaging
#
tbeg   = 6800.
tend   = 9600.
fldstp = 100
#
# case name (e.g., the Retau)
#
casename = '01E09'.zfill(5)
