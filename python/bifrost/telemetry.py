
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
import time
import uuid
import atexit
import socket
import inspect
import warnings
try:
    from urllib2 import urlopen
    from urllib import urlencode
except ImportError:
    from urllib.request import urlopen
    from urllib.parse import urlencode
from threading import RLock
from functools import wraps

import bifrost.version

# Create the cache directory
if not os.path.exists(os.path.join(os.path.expanduser('~'), '.bifrost')):
    os.mkdir(os.path.join(os.path.expanduser('~'), '.bifrost'))
_CACHE_DIR = os.path.join(os.path.expanduser('~'), '.bifrost', 'telemetry_cache')
if not os.path.exists(_CACHE_DIR):
    os.mkdir(_CACHE_DIR)

# Load the install ID key, creating it if it doesn't exist
_INSTALL_KEY = os.path.join(_CACHE_DIR, 'install.key')
if not os.path.exists(_INSTALL_KEY):
    with open(_INSTALL_KEY, 'w') as fh:
        fh.write(str(uuid.uuid4()))

with open(_INSTALL_KEY, 'r') as fh:
    _INSTALL_KEY = fh.read().rstrip()

# Reporting control
TELEMETRY_MAX_ENTRIES = 100
TELEMETRY_TIMEOUT     = 120   # s
TELEMETRY_ACTIVE      = True
_ACTIVE_KEY = os.path.join(_CACHE_DIR, 'do_not_report')
if os.path.exists(_ACTIVE_KEY):
    TELEMETRY_ACTIVE = False


class _TelemetryClient(object):
    """
    Bifrost telemetry client to help understand usage of the Bifrost Python interface.
    """
    _lock = RLock()
    
    def __init__(self, key, version=bifrost.version.__version__):
        # Setup
        self.key = key
        self.version = version
        
        # Session reference
        self._session_start = time.time()
        
        # Telemetry cache
        self._cache = {}
        self._cache_count = 0
        
        # Reporting lockout
        self.active = TELEMETRY_ACTIVE
        
        # Register the "send" method to be called by atexit... at exit
        atexit.register(self.send, True)
        
    def track(self, name, timing=0.0):
        """
        Add an entry to the telemetry cache with optional timing information.
        """
        
        if name[:7] != 'bifrost' or not self.active:
            return False
            
        with self._lock:
            try:
                self._cache[name][0] += 1
                self._cache[name][1] += (1 if timing > 0 else 0)
                self._cache[name][2] += timing
            except KeyError:
                self._cache[name] = [1, 0, timing]
                self._cache[name][1] += (1 if timing > 0 else 0)
                self._cache_count += 1
                
            if self._cache_count >= TELEMETRY_MAX_ENTRIES:
                self.send()
                
        return True
                
    def send(self, final=False):
        """
        Send the current cache of telemetry data back to the maintainers for 
        analysis.
        """
        
        success = False
        with self._lock:
            if self.active and self._cache_count > 0:
                try:
                    tNow = time.time()
                    payload = ';'.join(["%s;%i;%i;%.6f" % (name,
                                                           self._cache[name][0],
                                                           self._cache[name][1],
                                                           self._cache[name][2]) for name in self._cache])
                    payload = urlencode({'timestamp'   : int(tNow),
                                         'key'         : self.key, 
                                         'version'     : self.version,
                                         'session_time': "%.6f" % ((tNow-self._session_start) if final else 0.0,),
                                         'payload'     : payload})
                    try:
                        payload = payload.encode()
                    except AttributeError:
                        pass
                    uh = urlopen('https://fornax.phys.unm.edu/telemetry/bifrost.php', payload, 
                                 timeout=TELEMETRY_TIMEOUT)
                    status = uh.read()
                    if status == '':
                        self.clear()
                        success = True
                except Exception as e:
                    warnings.warn("Failed to send telemetry data: %s" % str(e))
            else:
                self.clear()
                
        return success
                
    def clear(self):
        """
        Clear the current telemetry cache.
        """
        
        with self._lock:
            self._cache.clear()
            self._cache_count = 0
            
    @property
    def is_active(self):
        """
        Whether or not the cache is active and sending data back.
        """
        
        return self.active
        
    def enable(self):
        """
        Enable saving data to the telemetry cache.
        """
        
        TELEMETRY_ACTIVE = True
        try:
            os.unlink(_ACTIVE_KEY)
        except OSError:
            pass
        self.active = TELEMETRY_ACTIVE
        
    def disable(self):
        """
        Disable saving data to the telemetry cache in a persistent way.
        """
        
        TELEMETRY_ACTIVE = False
        try:
            with open(_ACTIVE_KEY, 'w') as fh:
                fh.write('True')
        except OSError:
            pass
        self.active = TELEMETRY_ACTIVE


# Create an instance of the telemetry client to use.
_telemetry_client = _TelemetryClient(_INSTALL_KEY)


# Telemetry control
def is_active():
    """
    Return a boolean of whether or not the Bifrost telemetry client is active.
    """
    
    global _telemetry_client
    return _telemetry_client.is_active


def enable():
    """
    Enable logging of usage data via the Bifrost telemetry client.
    """
    
    global _telemetry_client
    _telemetry_client.enable()


def disable():
    """
    Disable logging of usage data via the Bifrost telemetry client.
    
    .. note::
        This function disables in a global way that persists across
        invocations.
    """
    
    global _telemetry_client
    _telemetry_client.disable()


def track_script():
    """
    Record the use of a Bifrost script.
    """
    
    global _telemetry_client
    
    caller = inspect.currentframe().f_back
    name = os.path.basename(caller.f_globals['__file__'])
    _telemetry_client.track('bifrost.tools.'+name)


def track_module():
    """
    Record the import of an Bifrost module.
    """
    
    global _telemetry_client
    
    caller = inspect.currentframe().f_back
    _telemetry_client.track(caller.f_globals['__name__'])


def track_function(user_function):
    """
    Record the use of a function in Bifrost without execution time information.
    """
    
    global _telemetry_client
    
    caller = inspect.currentframe().f_back
    mod = caller.f_globals['__name__']
    fnc = user_function.__name__
    name = mod+'.'+fnc+'()'
    
    @wraps(user_function)
    def wrapper(*args, **kwds):
        global _telemetry_client
        result =  user_function(*args, **kwds)
        
        _telemetry_client.track(name)
        return result
        
    return wrapper


def track_function_timed(user_function):
    """
    Record the use of a function in Bifrost with execution time information.
    """
    
    global _telemetry_client
    
    caller = inspect.currentframe().f_back
    mod = caller.f_globals['__name__']
    fnc = user_function.__name__
    name = mod+'.'+fnc+'()'
    
    @wraps(user_function)
    def wrapper(*args, **kwds):
        global _telemetry_client
        t0 = time.time()
        result = user_function(*args, **kwds)
        t1 = time.time()
        
        _telemetry_client.track(name, t1-t0)
        return result
        
    return wrapper


def track_method(user_method):
    """
    Record the use of a method in Bifrost with execution time information.
    """
    
    global _telemetry_client
    
    caller = inspect.currentframe().f_back
    mod = caller.f_globals['__name__']
    cls = None
    fnc = user_method.__name__
    name = mod+'.'+'%s'+'.'+fnc+'()'
    
    @wraps(user_method)
    def wrapper(*args, **kwds):
        global _telemetry_client
        result =  user_method(*args, **kwds)
        
        cls = type(args[0]).__name__
        _telemetry_client.track(name % cls)
        return result
        
    return wrapper


def track_method_timed(user_method):
    """
    Record the use of a method in Bifrost with execution time information.
    """
    
    global _telemetry_client
    
    caller = inspect.currentframe().f_back
    mod = caller.f_globals['__name__']
    cls = None
    fnc = user_method.__name__
    name = mod+'.'+'%s'+'.'+fnc+'()'
    
    @wraps(user_method)
    def wrapper(*args, **kwds):
        global _telemetry_client
        t0 = time.time()
        result =  user_method(*args, **kwds)
        t1 = time.time()
        
        cls = type(args[0]).__name__
        _telemetry_client.track(name % cls, t1-t0)
        return result
        
    return wrapper
