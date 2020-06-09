'''
Utility functions with no other good home.

'''

from collections import OrderedDict
from contextlib import contextmanager
import atexit
import copy
import functools
import sys
import time


def import_name(dotpath):
    '''
    Import a dotted-path name from the Python environment. Does not support
    top-level modules. Use ``__import__`` for that.

    Example:

        >>> import_name('time.time')
        <built-in function time>

    '''
    modulename, name = dotpath.rsplit('.', 1)
    module = __import__(modulename)
    parts = modulename.split('.')[1:]
    for part in parts:
        module = getattr(module, part)
    return getattr(module, name)


def get_subclass(cls, classname):
    '''
    Get a subclass for the named class. The ``classname`` must be the full
    dotted path to the subclass.

    Example:

        >>> class A(object): pass
        >>> class B(A): pass
        >>> get_subclass(A, '__main__.B')
        <class '__main__.B'>

    '''
    subclasses = list(cls.__subclasses__())
    while subclasses:
        cls = subclasses.pop()
        subclasses.extend(cls.__subclasses__())
        name = cls.__module__ + '.' + cls.__name__
        if name == classname:
            return cls
    raise ValueError


def merge_dict(dest, source):
    '''
    Similar to dict.update(), but descends into dict types and merges their
    keys recursively.

    Example:

        >>> a = {'foo': {'bar': 3, 'baz': 4}}
        >>> b = {'foo': {'baz': 5}}
        >>> merge_dict(a, b)
        {'foo': {'bar': 3, 'baz': 5}}

    '''
    result = copy.deepcopy(dest)
    for k, v in source.iteritems():
        if k not in result:
            result[k] = source[k]
        else:
            if isinstance(result[k], dict) and isinstance(source[k], dict):
                result[k] = merge_dict(result[k], source[k])
            else:
                result[k] = source[k]
    return result


def _wrap_constant(x):
    '''
    If ``x`` is callable, return ``x``. Otherwise, return a 1-arg callable
    which returns ``x`` unconditionally.

    Example:

        >>> f = _wrap_constant(3)
        >>> f(0)
        3

    '''
    return (x if hasattr(x, '__call__') else (lambda p: x))


class PrettyPartial(functools.partial):

    def __str__(self):
        keywords = self.keywords or {}
        args = ", ".join([repr(arg) for arg in self.args])
        kwargs = ", ".join(["%s=%r" % (k, v) for k, v in keywords.items()])
        parameters = [p for p in [args, kwargs] if p]
        parameters = ', '.join(parameters)
        return "{name}({parameters})".format(name=self.func.__name__,
                                             parameters=parameters)


class attrdict(dict):

    """ Maps getattr() calls to getitem().
    """

    def __getattr__(self, attr):
        return self[attr]

    def __setattr__(self, attr, value):
        self[attr] = value

    def __delattr__(self, attr):
        del self[attr]


def timeit(fn, *args, **kw):
    '''
    Time how much CPU time is taken to call `fn` with given args and keyword
    arguments.

    Example:

        >>> seconds, result = timeit(sum, [1, 2, 3])
        >>> print(result)
        6
        >>> seconds, result = timeit(time.leep 1)
        >>> print(seconds)
        1.0

    '''
    t0 = time.clock()
    result = fn(*args, **kw)
    t1 = time.clock()
    return (t1 - t0), result


@contextmanager
def profile(name):
    '''
    Profile sections of code with named counters. Multiple calls to the same
    counters are additive. Registers an ``atexit()`` call which shows the
    results at the end of the process. This measures wall time, not CPU time.

    Example:

        >>> with profile('sleep'): time.sleep(1)
        >>> with profile('sleep'): time.sleep(1)
        >>> sys.exit()
        sleep 2.0

    '''
    global _profile_registered
    if not _profile_registered:
        atexit.register(_profile_print_results)
        _profile_registered = True
    total = _profile_results.get(name, 0.)
    t0 = time.time()
    try:
        yield
    finally:
        _profile_results[name] = total + time.time() - t0
_profile_results = OrderedDict()
_profile_registered = False


def _profile_print_results():
    '''Helper function for ``profile()``'''
    if _profile_results:
        sys.stderr.write('\n')
    for name, value in _profile_results.items():
        sys.stderr.write('{} {}\n'.format(name, value))
