
# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
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

import unittest
import bifrost as bf
from socket import AF_INET, AF_INET6

class AddressTest(unittest.TestCase):
    def run_address_test(self, address, port, mtu, family=None,
                         check_address=True, check_family=False):
        addr = bf.address.Address(address, port, family)
        self.assertEqual(addr.port, port)
        self.assertGreater(addr.mtu, 0)
        self.assertLessEqual(addr.mtu, mtu)
        if check_address:
            self.assertEqual(addr.address, address)
            self.assertEqual(str(addr), "%s:%i" % (address, port))
        if check_family:
            self.assertEqual(addr.family, family)
    def test_localhost(self):
        self.run_address_test('127.0.0.1', 8123, mtu=65535)
    def test_zeros(self):
        self.run_address_test('0.0.0.0',   8123, mtu=65535)
    def test_eights(self):
        self.run_address_test('8.8.8.8',     80, mtu=1500)
    def test_google(self):
        self.run_address_test('google.com',  80, mtu=1500, family=AF_INET,
                              check_address=False, check_family=True)
    @unittest.skip("Connection doesn't work from Travis server")
    def test_google_IPv6(self):
        self.run_address_test('google.com',  80, mtu=1500, family=AF_INET6,
                              check_address=False, check_family=True)
