#!/usr/bin/env python

# Copyright (c) 2021, The Bifrost Authors. All rights reserved.
# Copyright (c) 2021, The University of New Mexico. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Python2 compatibility
from __future__ import print_function, division, absolute_import
    
import os
import sys
import argparse

from bifrost import telemetry
telemetry.track_script()


def main(args):
    # Toggle
    if args.enable:
        telemetry.enable()
    elif args.disable:
        telemetry.disable()
        
    # Report
    ## Status
    print("Bifrost Telemetry is %s" % ('active' if telemetry.is_active() else 'in-active'))
    
    ## Key
    if args.key:
        print("  Identification key: %s" % telemetry._INSTALL_KEY)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description='update the Bifrost telemetry setting', 
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
            )
    tgroup = parser.add_mutually_exclusive_group(required=False)
    tgroup.add_argument('-e', '--enable', action='store_true', 
                        help='enable telemetry for Bifrost')
    tgroup.add_argument('-d', '--disable', action='store_true', 
                        help='disable telemetry for Bifrost')
    parser.add_argument('-k', '--key', action='store_true',
                        help='show install identification key')
    args = parser.parse_args()
    main(args)
