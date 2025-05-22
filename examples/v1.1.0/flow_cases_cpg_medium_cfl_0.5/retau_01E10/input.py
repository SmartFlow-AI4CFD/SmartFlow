#!/usr/bin/python
# values should be consistent with dns.in
h     = 1.
ub    = 1.
visci = 6.00E+11
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
casename = '01E10'.zfill(5)
