#!/usr/bin/python
# values should be consistent with dns.in
h     = 1.
ub    = 1.
visci = 125000.
#
uconv = 0. # if we solve on a convective reference frame; else = 0.
#
# parameters for averaging
#
tbeg   = 3200.
tend   = 1800.
fldstp = 100
#
# case name (e.g., the Retau)
#
casename = '5200'.zfill(5)
