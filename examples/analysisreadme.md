# Description of feature
Check whether keys already exist in the default dicitonary. return error when a key in the dictionary doesn't exist in the default dictionary. this way we prevent the addition fo keys+values that don't exist.

# Potential solution
```python
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self

    def __setattr__(self, key, value):
        if key not in [*self.keys(), '__dict__']:
            raise KeyError('No new keys allowed')
        else:
            super().__setattr__(key, value)

    def __setitem__(self, key, value):
        if key not in self:
            raise KeyError('No new keys allowed')
        else:
            super().__setitem__(key, value)
```