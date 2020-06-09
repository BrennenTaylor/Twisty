'''
Configuration system.

'''

from util import get_subclass


class Element(object):
    '''
    Objects inheriting from this gain classmethods to help construct the object
    from Python dictionaries.

    For example:

        >>> class A(Element): pass
        >>> class B(Element): pass
        >>> Element.create({'type': '__main__.A'})
        <__main__.A object at 0x10f3c7450>
        >>> Element.create({'type': '__main__.B'})
        <__main__.B object at 0x10f3c7450>

    The magic happens in the `from_dict()` method which assumes that the passed
    dictionary is the keyword-arguments to the object's constructor. Subclasses
    may override `from_dict()` to use some custom behavior to construct the
    object.

    `Element.create()` assumes a key exists in the dictionary named `type`
    which specifies a subclass of Element to construct.
    '''

    @classmethod
    def get_type(cls):
        '''Get the Python dotted-path name of this object'''
        return cls.__module__ + '.' + cls.__name__

    @classmethod
    def create(cls, d):
        '''Create object from provided dictionary'''
        dotpath = d.pop('type')
        subclass = get_subclass(cls, dotpath)
        return subclass.from_dict(d)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

    def to_dict(self):
        raise NotImplementedError
