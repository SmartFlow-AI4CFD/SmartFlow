#!/usr/bin/python
# values should be consistent with dns.in
h     = 1.
ub    = 1.
visci = 20000.
#
uconv = 0. # if we solve on a convective reference frame; else = 0.
#
# parameters for averaging
#
tbeg   = 1600.
tend   = 3200.
fldstp = 100
#
# case name (e.g., the Retau)
#
casename = '1000'.zfill(5)
